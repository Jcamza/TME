"""Single-pass image inference of both missing Table 2 rows, across MOS datasets.

Run from NegLabel_Crop_Compare root:
    python -m table2_idonly.run --dataset all --out results/table2_idonly_alpha07
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .core import (DATASETS, DEFAULT_ROOTS, EXPECTED_COUNTS, PairedViews,
                   feature_fusion, l2_normalize, mcm_from_features, scan_images)

FIELDNAMES = ('path', 'mcm_global', 'mcm_fivecrop')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Paired MCM: ID-only global vs ID-only FiveCrop')
    p.add_argument('--dataset', nargs='+', choices=(*DATASETS, 'all'), default=['all'])
    p.add_argument('--out', type=Path, default=Path('results/table2_idonly_alpha07'))
    p.add_argument('--neglabel-root', type=Path, default=Path('/data/hx/code/NegLabel'))
    p.add_argument('--model', default='ViT-B/16',
                   help='Official OpenAI CLIP model name or matching local .pt checkpoint path')
    p.add_argument('--prompt-idx', type=int, default=85)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-samples', type=int, default=0, help='0 = all; positive = SMOKE ONLY')
    p.add_argument('--allow-count-mismatch', action='store_true',
                   help='Permit non-MOS counts; do not publish as comparable MOS results')
    p.add_argument('--check-only', action='store_true',
                   help='Check dataset files, official class names and prompt; no model GPU load')
    p.add_argument('--no-resume', action='store_true',
                   help='Delete partial scores and recompute this selected dataset')
    for dataset in DATASETS:
        p.add_argument('--root-' + dataset.lower(), type=Path,
                       default=Path(DEFAULT_ROOTS[dataset]), help=f'Image root for {dataset}')
    args = p.parse_args(argv)
    if args.dataset == ['all']:
        args.dataset = list(DATASETS)
    elif 'all' in args.dataset:
        p.error('Use --dataset all alone, or list individual dataset names')
    args.dataset = [d for d in DATASETS if d in args.dataset]
    if not 0 <= args.alpha <= 1:
        p.error('--alpha must be between 0 and 1')
    if args.batch_size <= 0 or args.workers < 0 or args.max_samples < 0:
        p.error('Invalid batch/worker/max-samples setting')
    return args


def sha256_lines(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def get_official_classes(neglabel_root: Path, prompt_idx: int):
    source = neglabel_root / 'mmcls/models/classifiers/class_names.py'
    if not source.is_file():
        raise FileNotFoundError(f'Official class_names.py missing: {source}')
    # runpy.run_path reads the exact official repository file, not an alternative ImageNet label list.
    ns = runpy.run_path(str(source))
    names = ns['CLASS_NAME']['imagenet']
    template = ns['prompt_templates'][prompt_idx]
    if len(names) != 1000:
        raise RuntimeError(f'Official ImageNet class list must have 1000 classes, found {len(names)}')
    prompts = [template.format(name) for name in names]
    return prompts, source


def load_clip_model(name, device):
    try:
        import clip
    except ImportError as e:
        raise RuntimeError('OpenAI CLIP package missing in the active iso environment') from e
    model, _ = clip.load(name, device=device, jit=False)
    model.eval()
    model.requires_grad_(False)
    return model, clip.tokenize


@torch.inference_mode()
def encode_positive_text(model, tokenizer, prompts, device, batch_size=256):
    out = []
    for start in range(0, len(prompts), batch_size):
        token = tokenizer(prompts[start:start+batch_size]).to(device)
        out.append(l2_normalize(model.encode_text(token)))
    U = torch.cat(out, dim=0)
    assert U.shape[0] == 1000
    return U


def make_fingerprint(args, dataset, root, paths, prompts):
    return {
        'dataset': dataset,
        'root': str(root.resolve()),
        'count': len(paths),
        'path_manifest_sha256': sha256_lines([p.relative_to(root).as_posix() for p in paths]),
        'prompts_sha256': sha256_lines(prompts),
        'model': args.model,
        'alpha': args.alpha,
        'prompt_idx': args.prompt_idx,
        'logit_scale': 100.0,
        'global_preprocess': 'Resize(short_side=224,BICUBIC),CenterCrop(224),RGB,CLIPNorm',
        'fivecrop_preprocess': 'Resize(short_side=256,BICUBIC),FiveCrop(224),RGB,CLIPNorm',
        'feature_fusion': 'L2((1-alpha)*L2(global)+alpha*mean(L2(five crops)))',
        'mcm': 'max(softmax(100*cosine,dim=classes))',
    }


def check_file_fingerprint(path: Path, fingerprint):
    info = json.loads(path.read_text(encoding='utf-8'))
    old = info['fingerprint']
    if old != fingerprint:
        differences = {k: (old.get(k), fingerprint.get(k)) for k in fingerprint if old.get(k) != fingerprint.get(k)}
        raise RuntimeError(f'Existing scores have different settings: {path}; differences={differences}. '
                           'Choose a different --out or use --no-resume.')
    return info


def inspect_existing_csv(csv_path: Path, paths, root: Path):
    """Validate saved rows are a strict, ordered path prefix (safe resume)."""
    n = 0
    if not csv_path.exists():
        return n
    with csv_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDNAMES):
            raise RuntimeError(f'Bad CSV schema in {csv_path}: {reader.fieldnames}')
        for row in reader:
            if n >= len(paths) or row['path'] != paths[n].relative_to(root).as_posix():
                raise RuntimeError(f'Existing CSV is not an ordered manifest prefix at row {n}: {csv_path}')
            for k in FIELDNAMES[1:]:
                val = float(row[k])
                if not np.isfinite(val) or not (0 <= val <= 1):
                    raise RuntimeError(f'Invalid {k} at row {n} in {csv_path}')
            n += 1
    return n


def perform_dataset(args, dataset, root, paths, model, U, device, prompts):
    args.out.mkdir(parents=True, exist_ok=True)
    final_csv = args.out / f'{dataset}.csv'
    partial_csv = args.out / f'{dataset}.partial.csv'
    meta_file = args.out / f'{dataset}.meta.json'
    fingerprint = make_fingerprint(args, dataset, root, paths, prompts)
    if args.no_resume:
        for file in (final_csv, partial_csv, meta_file):
            file.unlink(missing_ok=True)
    if meta_file.exists():
        check_file_fingerprint(meta_file, fingerprint)
    elif partial_csv.exists() or final_csv.exists():
        raise RuntimeError(f'Metadata missing for existing scores of {dataset}: refuse unsafe resume')
    else:
        meta_file.write_text(json.dumps({'fingerprint': fingerprint, 'complete': False}, indent=2),
                             encoding='utf-8')

    if final_csv.exists():
        n = inspect_existing_csv(final_csv, paths, root)
        if n != len(paths):
            raise RuntimeError(f'Final CSV incomplete ({n} / {len(paths)}): {final_csv}')
        print(f'[SKIP] {dataset}: {n} verified saved scores')
        return
    done = inspect_existing_csv(partial_csv, paths, root)
    if done >= len(paths):
        partial_csv.rename(final_csv)
        meta_file.write_text(json.dumps({'fingerprint': fingerprint, 'complete': True}, indent=2),
                             encoding='utf-8')
        return

    ds = PairedViews(root, paths[done:])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=device.type == 'cuda',
                    persistent_workers=args.workers > 0)
    with partial_csv.open('a' if done else 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not done:
            writer.writeheader()
        for relative_paths, global_batch, five_batch in tqdm(dl, desc=f'{dataset}: {done}/{len(paths)}', unit='batch'):
            global_batch = global_batch.to(device, non_blocking=True)
            five_batch = five_batch.to(device, non_blocking=True)
            B, C, channels, h, w = five_batch.shape
            with torch.inference_mode():
                h0 = l2_normalize(model.encode_image(global_batch))
                hcrop = l2_normalize(model.encode_image(five_batch.reshape(B*C, channels, h, w)))
                hcrop = hcrop.view(B, C, -1)
                hfused = feature_fusion(h0, hcrop, args.alpha)
                original = mcm_from_features(h0, U).cpu().numpy()
                mixed = mcm_from_features(hfused, U).cpu().numpy()
            for path, s0, s1 in zip(relative_paths, original, mixed):
                writer.writerow({'path': path, 'mcm_global': f'{float(s0):.10g}',
                                 'mcm_fivecrop': f'{float(s1):.10g}'})
            file.flush()  # enables safe batch-granular resume
    complete = inspect_existing_csv(partial_csv, paths, root)
    if complete != len(paths):
        raise RuntimeError(f'Expected {len(paths)} scores but only have {complete} in {partial_csv}')
    partial_csv.rename(final_csv)
    meta_file.write_text(json.dumps({'fingerprint': fingerprint, 'complete': True,
                                     'finished_utc': datetime.now(timezone.utc).isoformat()}, indent=2),
                         encoding='utf-8')
    print(f'[DONE] {dataset}: {complete} images, {final_csv}')


def main(argv=None):
    args = parse_args(argv)
    info = {}
    prompts, class_source = get_official_classes(args.neglabel_root, args.prompt_idx)
    print(f'[CONFIG] original official classes/prompt: {class_source} / index={args.prompt_idx}')
    print(f'[CONFIG] prompt example: {prompts[0]}')
    for name in args.dataset:
        root = getattr(args, 'root_' + name.lower())
        full_paths = scan_images(root)
        if len(full_paths) != EXPECTED_COUNTS[name] and not args.allow_count_mismatch:
            raise RuntimeError(f'{name}: found {len(full_paths)}, expected {EXPECTED_COUNTS[name]} MOS images. '
                               'Use --allow-count-mismatch only for non-paper tests.')
        paths = full_paths[:args.max_samples] if args.max_samples else full_paths
        info[name] = (root, paths)
        print(f'[DATA] {name:<12} {len(paths):>6} selected / {len(full_paths)} total images at {root}')
    if args.check_only:
        print('[PREFLIGHT OK] No model loading, no scores computed')
        return 0
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable, requested {device}')
    torch.manual_seed(42)
    model, tokenizer = load_clip_model(args.model, str(device))
    U = encode_positive_text(model, tokenizer, prompts, device)
    print(f'[TEXT] positive prototypes U = {tuple(U.shape)}; negative text not encoded')
    for name, (root, paths) in info.items():
        perform_dataset(args, name, root, paths, model, U, device, prompts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
