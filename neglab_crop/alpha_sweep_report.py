"""Summarize alpha-sweep results from the exact official NegLabel metric CSV.

No raw labels are used to choose alpha; the best observed alpha is exploratory
and must not be reported as a held-out, label-free hyperparameter choice.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_alpha_method(method):
    if method.startswith('cls_attention_'):
        m = re.fullmatch(r'cls_attention_(mask|bbox)_mass(\d+)_alpha(\d+(?:\.\d+)?)', method)
        return (f'CLS{m.group(2)} {m.group(1)}', float(m.group(3))) if m else None
    m = re.fullmatch(r'object_dino_bbox_alpha(\d+(?:\.\d+)?)', method)
    return ('Object-DINO bbox', float(m.group(1))) if m else None


def build_summary(comparison):
    with open(comparison, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    summary = []
    for r in rows:
        parsed = parse_alpha_method(r['method'])
        if parsed is None:
            continue
        branch, alpha = parsed
        summary.append({
            'dataset': r['dataset'], 'branch': branch, 'alpha': alpha,
            'AUROC': float(r['AUROC']), 'FPR95': float(r['FPR95']),
            'delta_AUROC': float(r['delta_AUROC_vs_original']),
            'delta_FPR95': float(r['delta_FPR95_vs_original']),
        })
    return summary


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results=build_summary(args.comparison)
    if not results:
        raise ValueError('No CLS or Object-DINO alpha entries found')
    out=args.output_dir/'alpha_sweep_summary.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped=defaultdict(list)
    for r in results:
        grouped[r['dataset']].append(r)
    for dataset,items in grouped.items():
        for metric, ylabel, invert in [
            ('delta_AUROC','Delta AUROC vs original (pp)',False),
            ('delta_FPR95','Delta FPR95 vs original (pp)',True)]:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for branch in sorted(set(r['branch'] for r in items)):
                vals=sorted((r for r in items if r['branch']==branch),key=lambda r:r['alpha'])
                if vals:
                    ax.plot([r['alpha'] for r in vals],[r[metric] for r in vals],
                            marker='o',label=branch)
            ax.axhline(0,linewidth=1,linestyle='--')
            ax.set_xlabel('Fusion weight alpha')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{dataset}: {metric} (lower is better)' if invert else
                         f'{dataset}: {metric} (higher is better)')
            ax.grid(alpha=.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.output_dir/f'alpha_curve_{dataset}_{metric}.png',dpi=165)
            plt.close(fig)
    print('[ALPHA-SWEEP] Summary:',out)
    print('[ALPHA-SWEEP] Curves:',len(grouped)*2)
    print('[NOTE] Per-dataset best test alpha is exploratory, NOT a label-free hyperparameter selection.')
    return results


if __name__=='__main__':
    main()
