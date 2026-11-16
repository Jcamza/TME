"""Paired NegLabel OOD experiment: original, FiveCrop, CLS-70 and Object-DINO.

Usage: python -m neglab_crop.attention_runner --help
This is an ADDITIONAL runner; the already validated runner.py is unmodified.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .attention_regions import (
    CLIPLastLayerAttention, DINOv2ObjectRegions, box_crops_as_clip,
    boxes_from_mask, masked_clip_image, select_attention_mass,
)
from .data import read_dataset_from_config
from .official import load_official
from .preprocess import ImageViews
from .runner import _encode, _official_score, _make_plot
from .scoring import feature_fusion


class RegionDataset(Dataset):
    """Return EXACT original NegLabel input and spatially aligned RGB tensor.

    The CLIP RGB tensor is normalized with the exact official global transform;
    the aligned raw tensor is used for masking/crops and DINOv2 proposals.
    """
    def __init__(self, samples):
        self.samples = samples
        self.aligned = T.Compose([T.Resize(224), T.CenterCrop(224)])
        self.to_clip = T.Normalize((.48145466,.4578275,.40821073),
                                   (.26862954,.26130258,.27577711))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s.path) as image:
                img = self.aligned(image.convert('RGB'))
                raw = TF.to_tensor(img)
        except Exception as exc:
            raise RuntimeError(f'Cannot open {s.path}: {exc}') from exc
        return self.to_clip(raw.clone()), raw, s.label, str(s.path)


class RegionCache:
    """SQLite cache of deterministic region proposals; commits every batch."""
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=60)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('CREATE TABLE IF NOT EXISTS regions '
                          '(key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.conn.commit()

    @staticmethod
    def key(method, path, settings):
        p = Path(path)
        stat = p.stat()
        raw = json.dumps([method, str(p.resolve()), stat.st_size, stat.st_mtime_ns, settings],
                         sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key):
        found = self.conn.execute('SELECT payload FROM regions WHERE key=?', (key,)).fetchone()
        return json.loads(found[0]) if found else None

    def set(self, key, val):
        self.conn.execute('INSERT OR REPLACE INTO regions (key,payload) VALUES (?,?)',
                          (key, json.dumps(val)))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Original, FiveCrop, CLIP-CLS70, Object-DINO-guided NegLabel')
    p.add_argument('--official-root', type=Path, default=Path('/data/hx/code/NegLabel'))
    p.add_argument('--config', default='custom_config/clip.py')
    p.add_argument('--output-dir', type=Path, default=Path('results/attention_regions_alpha07'))
    p.add_argument('--cache-db', type=Path, default=Path('data/attention_region_cache.sqlite'))
    p.add_argument('--dino-model', default='facebook/dinov2-base',
                   help='HF Dinov2Model local path or huggingface repo ID')
    p.add_argument('--allow-model-download', action='store_true',
                   help='Allow HuggingFace download of missing DINOv2 weights (default: local cache only)')
    p.add_argument('--dino-clusters', type=int, default=12,
                   help='Released Object-DINO code uses 12 KMeans clusters')
    p.add_argument('--dino-knn', type=int, default=12)
    p.add_argument('--dino-n-init', type=int, default=3,
                   help='Reduced from released source n_init=10 for speed; set 10 for stricter alignment')
    p.add_argument('--dino-proposal-batch', type=int, default=4)
    p.add_argument('--alpha', type=float, default=.7)
    p.add_argument('--cls-mass', type=float, default=.7)
    p.add_argument('--cls-view', choices=['mask','bbox'], default='mask')
    p.add_argument('--include-fivecrop', action='store_true', help='Add already validated FiveCrop alpha=0.7')
    p.add_argument('--skip-dino', action='store_true', help='Diagnostic CLS-only run, NOT full comparison')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--view-batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit-id', type=int, default=0)
    p.add_argument('--limit-ood', type=int, default=0)
    p.add_argument('--dataset', action='append', default=[])
    p.add_argument('--id-path', default='')
    p.add_argument('--id-ann', default='')
    p.add_argument('--ood-override', action='append', default=[])
    p.add_argument('--clip-weights', default='')
    p.add_argument('--check-official-parity', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='Allow overwriting comparison.csv')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--visualize-n', type=int, default=4,
                   help='Number of annotated examples per dataset (set 0 to disable)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args(argv)


def _get_cache_entries(cache, method, paths, key_settings):
    keys = [cache.key(method, p, key_settings) for p in paths]
    stored = [cache.get(k) for k in keys]
    return keys, stored


@torch.inference_mode()
def _collect(name, samples, args, ood_model, dino, cache, out_dir):
    loader = DataLoader(RegionDataset(samples), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    encoder = ood_model.classifier.model
    device = next(encoder.parameters()).device
    cls_capture = CLIPLastLayerAttention(encoder)
    five = ImageViews('fivecrop') if args.include_fivecrop else None
    scores = defaultdict(list)
    diagnostic = []
    (out_dir/'examples').mkdir(exist_ok=True)
    paths_all, labels_all = [], []
    start = time.perf_counter()
    try:
        for ib, (original_cpu, raw_cpu, labels, paths) in enumerate(loader):
            paths = list(paths)
            original = original_cpu.to(device, non_blocking=True)
            cls_capture.start()
            gh = _encode(encoder, original, args.view_batch_size)
            attention_map = cls_capture.finish()
            orig = _official_score(ood_model, gh)
            scores['official_original'].extend(orig.cpu().tolist())
            if ib == 0 and args.check_official_parity:
                exact = ood_model.forward(img=original)
                diff = (orig - exact).abs().max().item()
                print(f'[{name}] Official parity: max_abs_error={diff:.7g}', flush=True)
                if diff > 1e-4:
                    raise AssertionError(f'Baseline parity failed: {diff}')

            keep = select_attention_mass(attention_map, args.cls_mass)
            if args.cls_view == 'mask':
                cls_view = masked_clip_image(original, keep)
                cls_box = None
            else:
                cls_box = boxes_from_mask(keep, size=224)
                cls_view = box_crops_as_clip(raw_cpu, cls_box).to(device)
            cls_feature = _encode(encoder, cls_view, args.view_batch_size)
            cls_score = _official_score(ood_model,
                                        feature_fusion(gh, cls_feature[:,None,:], args.alpha))
            scores[f'cls_attention_{args.cls_view}_mass{int(args.cls_mass*100)}_alpha{args.alpha:.2f}'].extend(
                cls_score.cpu().tolist())

            if dino is not None:
                settings = {
                    'dino_model': args.dino_model, 'clusters': args.dino_clusters,
                    'neighbors': args.dino_knn, 'n_init': args.dino_n_init,
                    'version': 'qkv_head_kmeans_ncut_v1',
                }
                keys, stored = _get_cache_entries(cache, 'dino_bbox', paths, settings)
                miss = [i for i,x in enumerate(stored) if x is None]
                if miss:
                    # Region boxes generated from ORIGINAL aligned RGB, not CLIP patch features.
                    new = dino.boxes(raw_cpu[miss])
                    for j, region in zip(miss, new):
                        payload = {
                            'box': list(region.box), 'patch_fraction': region.patch_fraction,
                            'box_fraction': region.box_fraction,
                            'selected_heads': region.selected_heads,
                            'fallback': region.fallback,
                        }
                        cache.set(keys[j], payload)
                        stored[j] = payload
                    cache.commit()
                dino_boxes = [tuple(x['box']) for x in stored]
                dino_view = box_crops_as_clip(raw_cpu, dino_boxes).to(device)
                dh = _encode(encoder, dino_view, args.view_batch_size)
                ds = _official_score(ood_model, feature_fusion(gh, dh[:,None,:], args.alpha))
                scores[f'object_dino_bbox_alpha{args.alpha:.2f}'].extend(ds.cpu().tolist())

            if five is not None:
                # ImageViews produces the same global and FiveCrop views as previous runner.
                # The original cropped RGB is 224 already; FiveCrop is defined on the
                # pre-center-crop image; reload the unchanged source to preserve protocol.
                five_views = []
                for path in paths:
                    with Image.open(path) as im:
                        views = five(im.convert('RGB'))
                    five_views.append(views[1:])
                fv = torch.stack(five_views).to(device)
                bs, nc = fv.shape[:2]
                fh = _encode(encoder, fv.reshape(-1,*fv.shape[-3:]),
                             args.view_batch_size).reshape(bs,nc,-1)
                fs = _official_score(ood_model, feature_fusion(gh, fh, args.alpha))
                scores[f'fivecrop_alpha{args.alpha:.2f}'].extend(fs.cpu().tolist())

            for i,path in enumerate(paths):
                record = {'filename': path,
                          'cls_kept_patches': int(keep[i].sum().item()),
                          'cls_patch_fraction': float(keep[i].float().mean().item()),
                          'cls_region_box': str(cls_box[i]) if cls_box is not None else '',
                          'dino_box': '', 'dino_patch_fraction': '',
                          'dino_box_fraction': '', 'dino_selected_heads': '',
                          'dino_fallback': ''}
                if dino is not None:
                    x = stored[i]
                    record.update({'dino_box':str(tuple(x['box'])),
                                   'dino_patch_fraction': x['patch_fraction'],
                                   'dino_box_fraction': x['box_fraction'],
                                   'dino_selected_heads': x['selected_heads'],
                                   'dino_fallback':x['fallback']})
                diagnostic.append(record)
            # Auditable region visualizations: original, CLS-70 retained pixels,
            # and Object-DINO rectangle drawn over the same aligned image.
            for i in range(len(paths)):
                index = len(paths_all) + i
                if index >= args.visualize_n:
                    break
                base = TF.to_pil_image(raw_cpu[i]).convert('RGB')
                cls_img = base.copy()
                patch_mask = torch.nn.functional.interpolate(
                    keep[i:i+1,None].float(), size=(224,224),mode='nearest')[0,0]
                cls_pixels = torch.as_tensor(np.asarray(base).copy())
                gray = torch.tensor([int(round(255*x)) for x in (.48145466,.4578275,.40821073)],
                                    dtype=cls_pixels.dtype)
                cls_pixels[~patch_mask.cpu().bool()] = gray
                cls_img = Image.fromarray(cls_pixels.numpy())
                obj_img = base.copy()
                if dino is not None:
                    draw = ImageDraw.Draw(obj_img)
                    draw.rectangle(dino_boxes[i], outline=(255,48,48), width=3)
                canvas = Image.new('RGB',(224*3,250),(250,250,250))
                canvas.paste(base,(0,24)); canvas.paste(cls_img,(224,24))
                canvas.paste(obj_img,(448,24))
                draw=ImageDraw.Draw(canvas)
                draw.text((8,6),'Original center crop',fill=(20,20,20))
                draw.text((232,6),f'CLIP CLS mass {args.cls_mass:.0%}',fill=(20,20,20))
                draw.text((456,6),'Object-DINO bbox',fill=(20,20,20))
                canvas.save(out_dir/'examples'/f'{name}_{index:04d}.jpg',quality=92)
            paths_all.extend(paths)
            labels_all.extend([int(i) for i in labels])
            print(f'[{name}] {len(paths_all)}/{len(samples)} completed', end='\r', flush=True)
    finally:
        cls_capture.close()
    print(f'[{name}] {len(paths_all)} images completed in {time.perf_counter()-start:.1f}s', flush=True)
    assert all(len(v)==len(paths_all) for v in scores.values())
    with (out_dir/'scores'/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['filename','gt_label',*scores.keys()])
        for i,p in enumerate(paths_all):
            w.writerow([p,labels_all[i],*(f'{scores[k][i]:.9f}' for k in scores)])
    with (out_dir/'diagnostics'/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostic[0]))
        w.writeheader(); w.writerows(diagnostic)
    summary = {
        'dataset': name,
        'images': len(diagnostic),
        'mean_cls_kept_patches': float(np.mean([d['cls_kept_patches'] for d in diagnostic])),
        'mean_cls_patch_fraction': float(np.mean([d['cls_patch_fraction'] for d in diagnostic])),
    }
    if dino is not None:
        summary.update({
            'mean_dino_box_fraction': float(np.mean([d['dino_box_fraction'] for d in diagnostic])),
            'dino_full_box_rate': float(np.mean([d['dino_box_fraction'] >= .95 for d in diagnostic])),
            'dino_fallback_rate': float(np.mean([bool(d['dino_fallback']) for d in diagnostic])),
            'dino_fallback_reasons': dict((reason, sum(1 for d in diagnostic if d['dino_fallback']==reason))
                                          for reason in sorted(set(d['dino_fallback'] for d in diagnostic)) if reason),
        })
        print(f"[{name}] Object-DINO fallback rate={summary['dino_fallback_rate']:.3%}; "
              f"mean box area={summary['mean_dino_box_fraction']:.3%}", flush=True)
    (out_dir/'diagnostics'/f'{name}_summary.json').write_text(
        json.dumps(summary,indent=2),encoding='utf-8')
    return {k:np.asarray(v,dtype=np.float64) for k,v in scores.items()}


def main(argv=None):
    args=parse_args(argv)
    if not (0<=args.alpha<=1 and 0<args.cls_mass<=1):
        raise ValueError('alpha must be in [0,1] and cls-mass in (0,1]')
    if args.batch_size<1 or args.view_batch_size<1 or args.workers<0:
        raise ValueError('Bad batch/view/worker count')
    if not torch.cuda.is_available():
        raise RuntimeError('Official NegLabel requires CUDA')
    if args.output_dir.joinpath('comparison.csv').exists() and not args.overwrite:
        raise FileExistsError(f'{args.output_dir}/comparison.csv already exists. Use another output folder.')
    os.environ.setdefault('LOCAL_RANK','0')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=args.official_root.expanduser().resolve()
    cfg,ood=load_official(root,args.config,arch_override=args.clip_weights or None)
    encoder=ood.classifier.model
    if not hasattr(encoder.visual,'transformer'):
        raise RuntimeError('CLIP CLS-attention experiment requires ViT encoder')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ('scores','diagnostics'):
        (args.output_dir/name).mkdir(parents=True,exist_ok=True)
    selected=set(args.dataset)
    choices=[(x['name'],dict(x)) for x in cfg.data.ood_data if not selected or x['name'] in selected]
    if selected-set(x[0] for x in choices):
        raise ValueError(f'Unknown datasets: {selected-set(x[0] for x in choices)}')
    id_spec=dict(cfg.data.id_data)
    if args.id_path:id_spec['path']=args.id_path
    if args.id_ann:id_spec['data_ann']=args.id_ann
    overrides={}
    for token in args.ood_override:
        if '=' not in token:
            raise ValueError(f'Expected NAME=PATH: {token}')
        name,path=token.split('=',1)
        overrides[name]=path
    if set(overrides)-set(x[0] for x in choices):
        raise ValueError('Unexpected OOD override(s): '+str(set(overrides)-set(x[0] for x in choices)))
    id_samples=read_dataset_from_config(root,id_spec,limit=args.limit_id)
    other=[]
    for name,spec in choices:
        if name in overrides:spec['path']=overrides[name]
        other.append((name,read_dataset_from_config(root,spec,limit=args.limit_ood)))
    params={'datetime_utc':datetime.now(timezone.utc).isoformat(),
            'methods':['official_original',f'cls_attention_{args.cls_view}_mass{int(args.cls_mass*100)}_alpha{args.alpha:.2f}',
                       *([] if args.skip_dino else [f'object_dino_bbox_alpha{args.alpha:.2f}']),
                       *([f'fivecrop_alpha{args.alpha:.2f}'] if args.include_fivecrop else [])],
            'id_count':len(id_samples),'ood_counts':{name:len(s) for name,s in other},
            'alpha':args.alpha,'cls_mass':args.cls_mass,'cls_view':args.cls_view,
            'include_fivecrop':args.include_fivecrop,'dino_model':None if args.skip_dino else args.dino_model,
            'dino_clusters':args.dino_clusters,'dino_n_init':args.dino_n_init,
            'dino_knn':args.dino_knn,'seed':args.seed,
            'partial_evaluation':bool(args.limit_id or args.limit_ood),
            'note':'Object-DINO-guided region adaptation; not identical to official TokenCut evaluation.'}
    (args.output_dir/'settings.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    print(json.dumps(params,indent=2),flush=True)
    device=next(encoder.parameters()).device
    dino=None
    cache=RegionCache(args.cache_db)
    try:
        if not args.skip_dino:
            print('[MODEL] Loading DINOv2 backbone:',args.dino_model,flush=True)
            dino=DINOv2ObjectRegions(args.dino_model,device=device,
                clusters=args.dino_clusters,graph_neighbors=args.dino_knn,
                n_init=args.dino_n_init,local_only=not args.allow_model_download,
                max_batch=args.dino_proposal_batch)
        id_scores=_collect('ID',id_samples,args,ood,dino,cache,args.output_dir)
        ood_scores={}
        for name,samples in other:
            ood_scores[name]=_collect(name,samples,args,ood,dino,cache,args.output_dir)
    finally:
        if dino is not None:dino.close()
        cache.close()
    from mmcls.utils.ood_evaluate import evaluate_all
    rows=[]
    for name,ood_sc in ood_scores.items():
        for method in id_scores:
            auroc,aprin,aprout,fpr=evaluate_all(id_scores[method],ood_sc[method])
            rows.append({'dataset':name,'method':method,
                        'AUROC':100*auroc,'AUPR_IN':100*aprin,'AUPR_OUT':100*aprout,
                        'FPR95':100*fpr,'n_id':len(id_samples),'n_ood':len(ood_sc[method]),
                        'delta_AUROC_vs_original':0.,'delta_FPR95_vs_original':0.})
        base=next(r for r in rows if r['dataset']==name and r['method']=='official_original')
        for r in rows:
            if r['dataset']==name:
                r['delta_AUROC_vs_original']=r['AUROC']-base['AUROC']
                r['delta_FPR95_vs_original']=r['FPR95']-base['FPR95']
    for method in id_scores:
        v=[r for r in rows if r['method']==method]
        rows.append({'dataset':'AVERAGE','method':method,
                     **{key:float(np.mean([r[key] for r in v])) for key in
                        ('AUROC','AUPR_IN','AUPR_OUT','FPR95','delta_AUROC_vs_original','delta_FPR95_vs_original')},
                     'n_id':len(id_samples),'n_ood':sum(r['n_ood'] for r in v)})
    with (args.output_dir/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (args.output_dir/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if not args.no_plot:_make_plot(rows,args.output_dir)
    print('\n====== PAIRED ATTENTION-REGION RESULTS (percentage) ======',flush=True)
    for r in rows:
        print(f"{r['dataset']:12s} {r['method']:38s} AUROC={r['AUROC']:6.2f} "
              f"FPR95={r['FPR95']:6.2f} dAUROC={r['delta_AUROC_vs_original']:+6.2f} "
              f"dFPR95={r['delta_FPR95_vs_original']:+6.2f}",flush=True)
    print('[SAVED]',args.output_dir,flush=True)
    if args.limit_id or args.limit_ood:
        print('[WARNING] PARTIAL EVALUATION; do not compare to official paper tables.',flush=True)
    return rows


if __name__=='__main__':
    main()
