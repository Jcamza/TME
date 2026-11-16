"""Evaluate original NegLabel and cropped-feature NegLabel in the SAME process.

IMPORTANT: production baseline and modified score call the official `grouping()`
method on the same official text embeddings, same model, same original images.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ViewsDataset, read_dataset_from_config
from .official import load_official
from .preprocess import ImageViews
from .scoring import feature_fusion, paired_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Official NegLabel original vs multi-crop paired OOD evaluation')
    p.add_argument('--official-root', type=Path, required=True, help='Unchanged tmlr-group/NegLabel checkout')
    p.add_argument('--config', default='custom_config/clip.py', help='Config relative to official-root')
    p.add_argument('--output-dir', type=Path, default=Path('results/crop_compare'))
    p.add_argument('--alpha', type=float, default=0.2)
    p.add_argument('--alpha-grid', default='', help='Optional: comma-separated fusion weights, e.g. 0,0.1,0.2,0.3')
    p.add_argument('--crop-mode', choices=['fivecrop', 'tencrop'], default='fivecrop')
    p.add_argument('--crop-resize', type=int, default=256)
    p.add_argument('--crop-size', type=int, default=224)
    p.add_argument('--fusion', choices=['feature', 'score', 'both'], default='feature')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--view-batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit-id', type=int, default=0, help='0 = full ID set (smoke only otherwise)')
    p.add_argument('--limit-ood', type=int, default=0, help='0 = full OOD set')
    p.add_argument('--dataset', action='append', default=[], help='OOD dataset names to include; repeatable')
    p.add_argument('--recursive-ood', action='store_true')
    p.add_argument('--clip-weights', default='', help='Optional local OpenAI CLIP .pt model override')
    p.add_argument('--id-path', default='', help='Optional override ID image directory from official config')
    p.add_argument('--id-ann', default='', help='Optional override ID TXT annotation path')
    p.add_argument('--ood-override', action='append', default=[], metavar='NAME=PATH',
                   help='Optional override OOD image directory; repeatable')
    p.add_argument('--check-official-parity', action='store_true',
                   help='Assert original path matches original ScalableClassifier.forward() on first batch')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args(argv)


def _normalize_like_official(raw):
    """Normalize in original encoder output dtype BEFORE fp32 cast (official behavior)."""
    h = raw / raw.norm(dim=-1, keepdim=True)
    return h.float()


@torch.no_grad()
def _encode(model, images, view_batch_size):
    parts = []
    for x in images.split(view_batch_size, dim=0):
        h = model.encode_image(x)
        parts.append(_normalize_like_official(h))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _similarities(text_model, features):
    p = 100.0 * features @ text_model.text_features_pos.float().T
    n = 100.0 * features @ text_model.text_features_neg.float().T
    return p, n


@torch.no_grad()
def _official_score(ood_model, features):
    p, n = _similarities(ood_model.classifier, features)
    if int(ood_model.ngroup) <= 1:
        raise ValueError('The official ngroup<=1 branch is not a valid sum-softmax score. Use ngroup>1.')
    if n.shape[1] < ood_model.ngroup:
        raise ValueError('Not enough negative labels to form groups')
    return ood_model.grouping(p, n, num=ood_model.group_fuse_num,
                              ngroup=ood_model.ngroup, random_permute=True)


def _alpha_list(args):
    s = [x.strip() for x in args.alpha_grid.split(',')] if args.alpha_grid.strip() else [str(args.alpha)]
    alpha = []
    for val in s:
        x = float(val)
        if not 0 <= x <= 1:
            raise ValueError(f'alpha must be in [0,1], not {x}')
        if x not in alpha:
            alpha.append(x)
    return alpha


def _method_name(fusion, alpha):
    return f'{fusion}_alpha_{alpha:.3f}'


@torch.no_grad()
def _collect_one(name, samples, image_views, ood_model, alphas, args, output_dir):
    data = ViewsDataset(samples, image_views)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    encoder = ood_model.classifier.model
    device = next(encoder.parameters()).device
    observed = defaultdict(list)
    paths, labels = [], []
    first = True
    started = time.perf_counter()
    for batch_index, (views, targets, filenames) in enumerate(loader):
        batch_sz, num_views = views.shape[:2]
        # No change to original central crop or its features.
        original = views[:, 0].to(device, non_blocking=True)
        global_h = _encode(encoder, original, args.view_batch_size)
        orig_score = _official_score(ood_model, global_h)
        observed['official_original'].extend(orig_score.cpu().tolist())
        if first and args.check_official_parity:
            official_direct = ood_model.forward(img=original)
            diff = (orig_score - official_direct).abs().max().item()
            if diff > 1e-4:
                raise AssertionError(f'Original path deviates from official forward: max={diff:.7g}')
            print(f'[{name}] Official parity on batch 0: max_abs_error={diff:.7g}', flush=True)
        first = False

        cropped = views[:, 1:].reshape(-1, *views.shape[-3:]).to(device, non_blocking=True)
        local_h = _encode(encoder, cropped, args.view_batch_size).reshape(batch_sz, num_views - 1, -1)
        if args.fusion in ('feature', 'both'):
            for alpha in alphas:
                h = feature_fusion(global_h, local_h, alpha)
                observed[_method_name('feature', alpha)].extend(_official_score(ood_model, h).cpu().tolist())
        if args.fusion in ('score', 'both'):
            # Scores of individual crops are averaged, then fused with global score.
            crop_scores = torch.stack([_official_score(ood_model, local_h[:, r, :])
                                       for r in range(num_views - 1)], dim=1).mean(dim=1)
            for alpha in alphas:
                s = (1.0 - alpha) * orig_score + alpha * crop_scores
                observed[_method_name('score', alpha)].extend(s.cpu().tolist())
        paths.extend(list(filenames))
        labels.extend(int(x) for x in targets)
        print(f'[{name}] {len(paths)}/{len(data)} images', end='\r', flush=True)
    print(f'[{name}] {len(paths)} images completed in {time.perf_counter()-started:.1f}s', flush=True)
    assert all(len(v) == len(paths) for v in observed.values())
    score_dir = output_dir / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    with (score_dir / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
        cols = ['filename', 'gt_label', *observed]
        w = csv.writer(f)
        w.writerow(cols)
        for i, filename in enumerate(paths):
            w.writerow([filename, labels[i], *(f'{observed[k][i]:.9f}' for k in observed)])
    return {k: np.asarray(v, dtype=np.float64) for k, v in observed.items()}, labels


def _make_plot(rows, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = list(dict.fromkeys(r['dataset'] for r in rows if r['dataset'] != 'AVERAGE'))
    methods = list(dict.fromkeys(r['method'] for r in rows))
    if not datasets or not methods:
        return
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(9, 2.0*n + 3), 5))
    x = np.arange(n)
    width = 0.8 / len(methods)
    for j, method in enumerate(methods):
        lookup = {(r['dataset'], r['method']): r for r in rows}
        vals = [lookup[(d, method)]['AUROC'] for d in datasets]
        ax.bar(x - .4 + width * (j + .5), vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 100)
    ax.set_ylabel('AUROC (%)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / 'auroc_comparison.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(9, 2.0*n + 3), 5))
    for j, method in enumerate(methods):
        vals = [lookup[(d, method)]['FPR95'] for d in datasets]
        ax.bar(x - .4 + width * (j + .5), vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 100)
    ax.set_ylabel('FPR95 (%)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / 'fpr95_comparison.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('Official NegLabel CLIPScalableClassifier hardcodes CUDA; run with a CUDA-enabled GPU.')
    os.environ.setdefault('LOCAL_RANK', '0')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    alphas = _alpha_list(args)
    official_root = args.official_root.expanduser().resolve()
    # Load original model and its original prompts, selected negatives, and group scorer.
    cfg, ood_model = load_official(official_root, args.config,
                                   arch_override=args.clip_weights or None)
    if args.crop_size != 224:
        raise ValueError('For an exact ViT-B/16 comparison, crop_size must be 224')
    if args.batch_size < 1 or args.view_batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid batch/view batch/workers')
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    view_transform = ImageViews(args.crop_mode, args.crop_resize, args.crop_size)
    name_filter = set(args.dataset)
    choices = [(s['name'], s) for s in cfg.data.ood_data if not name_filter or s['name'] in name_filter]
    if not choices:
        raise ValueError(f'No requested OOD datasets found: {sorted(name_filter)}')
    id_spec = dict(cfg.data.id_data)
    if args.id_path:
        id_spec['path'] = args.id_path
    if args.id_ann:
        id_spec['data_ann'] = args.id_ann
    overrides = {}
    for token in args.ood_override:
        if '=' not in token:
            raise ValueError(f'--ood-override expects NAME=PATH, got {token}')
        name, path = token.split('=', 1)
        overrides[name] = path
    known_names = {name for name, _ in choices}
    if set(overrides) - known_names:
        raise ValueError(f'Unknown OOD overrides: {sorted(set(overrides)-known_names)}')
    id_samples = read_dataset_from_config(official_root, id_spec, limit=args.limit_id)
    ood_samples = []
    for name, spec in choices:
        item = dict(spec)
        if name in overrides:
            item['path'] = overrides[name]
        ood_samples.append((name, read_dataset_from_config(official_root, item,
                            recursive=args.recursive_ood, limit=args.limit_ood)))
    params = {
        'datetime_utc': datetime.now(timezone.utc).isoformat(),
        'official_root': str(official_root),
        'config': str(args.config),
        'encoder_arch': str(ood_model.classifier.model.visual.__class__.__name__),
        'prompt_idx_pos': cfg.model.classifier.get('prompt_idx_pos'),
        'prompt_idx_neg': cfg.model.classifier.get('prompt_idx_neg'),
        'pencentile_official': cfg.model.classifier.get('pencentile'),
        'neg_topk_official': cfg.model.classifier.get('neg_topk'),
        'num_id_labels': int(ood_model.classifier.text_features_pos.shape[0]),
        'num_negative_labels': int(ood_model.classifier.text_features_neg.shape[0]),
        'ngroup_official': int(ood_model.ngroup),
        'group_fuse_num_official': ood_model.group_fuse_num,
        'temperature_official_t': float(ood_model.t),
        'alphas': alphas, 'crop_mode': args.crop_mode, 'crop_resize': args.crop_resize,
        'crop_size': args.crop_size, 'fusion': args.fusion, 'seed': args.seed,
        'id_count': len(id_samples), 'ood_counts': {n: len(s) for n, s in ood_samples},
        'partial_evaluation': bool(args.limit_id or args.limit_ood),
        'data_overrides': {'id_path': args.id_path, 'id_ann': args.id_ann,
                           'ood_override': args.ood_override},
        'note': 'Original and crop branches share SAME official pretrained CLIP, texts, NegMining, and grouping.',
    }
    (args.output_dir / 'settings.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    print(json.dumps(params, indent=2), flush=True)
    id_scores, id_labels = _collect_one('ID', id_samples, view_transform, ood_model,
                                        alphas, args, args.output_dir)
    all_scores = {}
    for name, samples in ood_samples:
        scores, _ = _collect_one(name, samples, view_transform, ood_model, alphas, args, args.output_dir)
        all_scores[name] = scores
    from mmcls.utils.ood_evaluate import evaluate_all  # exact OFFICIAL metric implementation
    rows = []
    for dataset_name, ood in all_scores.items():
        for method in id_scores:
            # Official returns fractional metrics; convert to percentages for tables.
            auroc, aupr_in, aupr_out, fpr95 = evaluate_all(id_scores[method], ood[method])
            metrics = {'AUROC': auroc*100., 'AUPR_IN': aupr_in*100.,
                       'AUPR_OUT': aupr_out*100., 'FPR95': fpr95*100.}
            rows.append({'dataset': dataset_name, 'method': method, **metrics,
                         'n_id': len(id_samples), 'n_ood': len(ood[method]),
                         'delta_AUROC_vs_original': '', 'delta_FPR95_vs_original': ''})
        baseline = next(r for r in rows if r['dataset'] == dataset_name and r['method'] == 'official_original')
        for row in rows:
            if row['dataset'] == dataset_name:
                row['delta_AUROC_vs_original'] = row['AUROC'] - baseline['AUROC']
                row['delta_FPR95_vs_original'] = row['FPR95'] - baseline['FPR95']
    methods = list(id_scores)
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        rows.append({'dataset': 'AVERAGE', 'method': method,
                     'AUROC': float(np.mean([r['AUROC'] for r in subset])),
                     'AUPR_IN': float(np.mean([r['AUPR_IN'] for r in subset])),
                     'AUPR_OUT': float(np.mean([r['AUPR_OUT'] for r in subset])),
                     'FPR95': float(np.mean([r['FPR95'] for r in subset])),
                     'n_id': len(id_samples), 'n_ood': sum(r['n_ood'] for r in subset),
                     'delta_AUROC_vs_original': float(np.mean([r['delta_AUROC_vs_original'] for r in subset])),
                     'delta_FPR95_vs_original': float(np.mean([r['delta_FPR95_vs_original'] for r in subset]))})
    with (args.output_dir / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    if not args.no_plot:
        _make_plot(rows, args.output_dir)
    print('\n========== PAIRED RESULTS (percentage) ==========')
    for r in rows:
        print(f"{r['dataset']:12s} {r['method']:24s} AUROC={r['AUROC']:6.2f} "
              f"FPR95={r['FPR95']:6.2f} dAUROC={r['delta_AUROC_vs_original']:+6.2f} "
              f"dFPR95={r['delta_FPR95_vs_original']:+6.2f}")
    print('Saved:', args.output_dir)
    if args.limit_id or args.limit_ood:
        print('WARNING: PARTIAL evaluation. Results are NOT comparable to official paper Table 1.')
    return rows


if __name__ == '__main__':
    main()
