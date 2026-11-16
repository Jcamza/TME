"""Compute official NegLabel metrics for the two new MCM rows, then merge two existing rows.

The reference CSV shipped with this package reproduces only rounded values from
NegLabel_PositiveNoisy.md. Replace it with *actual* baseline metrics if available.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import json
import math

import numpy as np

from .core import EXPECTED_COUNTS, OOD_DATASETS

METHODS = [
    ('ID-only MCM', 'ID-only', 'Global', 'measured'),
    ('ID-only + FiveCrop', 'ID-only', 'FiveCrop', 'measured'),
    ('NegLabel', 'NegLabel', 'Global', 'reference'),
    ('Ours', 'NegLabel', 'FiveCrop', 'reference'),
]
METRIC_NAMES = ('auroc', 'aupr_in', 'aupr_out', 'fpr95')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/table2_idonly_alpha07'))
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--neglabel-root', type=Path, default=Path('/data/hx/code/NegLabel'))
    p.add_argument('--reference', type=Path,
                   default=Path('table2_idonly/reference_metrics_from_handoff.csv'))
    p.add_argument('--allow-incomplete', action='store_true', help='Only for smoke tests')
    args = p.parse_args(argv)
    args.output = args.output or args.input
    return args


def load_official_evaluator(root: Path):
    path = root / 'mmcls/utils/ood_evaluate.py'
    if not path.is_file():
        raise FileNotFoundError(f'Official evaluation code missing: {path}')
    spec = importlib.util.spec_from_file_location('_table2_official_ood_evaluate', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_all


def read_score_csv(path: Path, allow_incomplete=False):
    if not path.is_file():
        raise FileNotFoundError(f'Missing scores: {path}')
    metadata_path = path.with_suffix('.meta.json')
    if not metadata_path.exists():
        raise RuntimeError(f'Metadata missing for {path}')
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not meta.get('complete'):
        raise RuntimeError(f'Run not marked complete: {metadata_path}')
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['path', 'mcm_global', 'mcm_fivecrop']:
            raise RuntimeError(f'Wrong score schema: {path} {reader.fieldnames}')
        rows = list(reader)
    if len(rows) != meta['fingerprint']['count']:
        raise RuntimeError(f'CSV count disagrees with metadata: {path}')
    if not allow_incomplete and len(rows) != EXPECTED_COUNTS[meta['fingerprint']['dataset']]:
        raise RuntimeError(f'Incomplete MOS benchmark: {path}: {len(rows)} samples')
    v0 = np.asarray([float(r['mcm_global']) for r in rows], dtype=np.float64)
    v1 = np.asarray([float(r['mcm_fivecrop']) for r in rows], dtype=np.float64)
    if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))):
        raise RuntimeError(f'Nonfinite MCM scores: {path}')
    return (v0, v1), meta


def read_reference(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing baseline reference CSV: {path}')
    result = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            k = (row['method'], row['dataset'])
            if k in result:
                raise RuntimeError(f'Duplicate reference key {k}')
            if row['method'] not in ('NegLabel', 'Ours') or row['dataset'] not in (*OOD_DATASETS, 'Average'):
                raise RuntimeError(f'Unknown reference method/dataset {k}')
            result[k] = {'auroc': float(row['auroc']),
                         'fpr95': float(row['fpr95']),
                         'source': row.get('source') or 'external',
                         'aupr_in': None, 'aupr_out': None}
    required = {(method, dataset) for method in ('NegLabel', 'Ours') for dataset in (*OOD_DATASETS, 'Average')}
    if set(result) != required:
        raise RuntimeError(f'Reference CSV missing entries: {sorted(required - set(result))}')
    return result


def write_csv(path: Path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') if row.get(k, '') is not None else '' for k in fields})


def latex_table(all_metrics):
    names = ['iNaturalist', 'SUN', 'Places', 'Textures', 'Average']
    body = [r'\begin{table*}[t]', r'\centering',
            r'\caption{Contributions of textual and visual evidence to zero-shot OOD detection. '
            r'AUROC$\uparrow$ / FPR95$\downarrow$ (\%). ID-only uses MCM; '
            r'NegLabel uses the original negative-label scoring.}',
            r'\label{tab:text_view_evidence}', r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{lllccccc}', r'\hline',
            r'Text evidence & View evidence & Method & iNaturalist & SUN & Places & Textures & Average \\',
            r'\hline']
    for method, text, vision, _ in METHODS:
        cells = []
        for ds in names:
            m = all_metrics[(method, ds)]
            cells.append(f"{m['auroc']:.2f} / {m['fpr95']:.2f}")
        method_txt = r'\textbf{Ours}' if method == 'Ours' else method
        body.append(f"{text} & {vision} & {method_txt} & " + ' & '.join(cells) + r' \\')
    body += [r'\hline', r'\end{tabular}}', r'\end{table*}', '']
    return '\n'.join(body)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = load_official_evaluator(args.neglabel_root)
    reference = read_reference(args.reference)
    id_scores, id_meta = read_score_csv(args.input / 'ID.csv', args.allow_incomplete)
    all_scores = {'ID': id_scores}
    for name in OOD_DATASETS:
        scores, meta = read_score_csv(args.input / f'{name}.csv', args.allow_incomplete)
        # Ensure same prompt/prototype/CLIP/alpha regardless of dataset.
        for key in ('prompts_sha256', 'model', 'alpha', 'prompt_idx', 'logit_scale'):
            if meta['fingerprint'][key] != id_meta['fingerprint'][key]:
                raise RuntimeError(f'{name}: incompatible setting {key}; refuse summary')
        all_scores[name] = scores
    measured = {}
    for i, method in enumerate(('ID-only MCM', 'ID-only + FiveCrop')):
        rows = []
        for name in OOD_DATASETS:
            values = scorer(all_scores['ID'][i], all_scores[name][i])
            row = dict(zip(METRIC_NAMES, [100.0 * float(s) for s in values]))
            row['source'] = 'measured:official_evaluate_all'
            measured[(method, name)] = row
            rows.append(row)
        measured[(method, 'Average')] = {
            key: float(np.mean([r[key] for r in rows])) for key in METRIC_NAMES
        } | {'source': 'measured:unrounded_macro_mean'}
    all_metrics = dict(measured)
    all_metrics.update(reference)
    ordered = []
    for method, text, vision, _ in METHODS:
        for name in (*OOD_DATASETS, 'Average'):
            values = all_metrics[(method, name)]
            ordered.append({'method': method, 'text': text, 'vision': vision, 'dataset': name,
                            'auroc': values['auroc'], 'fpr95': values['fpr95'],
                            'aupr_in': values.get('aupr_in'), 'aupr_out': values.get('aupr_out'),
                            'source': values['source'], 'n_id': len(id_scores[0]) if method.startswith('ID-only') else None,
                            'n_ood': len(all_scores[name][0]) if name != 'Average' and method.startswith('ID-only') else None})
    output = args.output / 'table2_long.csv'
    write_csv(output, ordered, ['method', 'text', 'vision', 'dataset', 'auroc', 'fpr95',
                                'aupr_in', 'aupr_out', 'source', 'n_id', 'n_ood'])
    (args.output / 'table2_paper.tex').write_text(latex_table(all_metrics), encoding='utf-8')
    deltas = []
    comparisons = [
        ('view_only_mcm', 'ID-only MCM', 'ID-only + FiveCrop'),
        ('view_with_negative_text', 'NegLabel', 'Ours'),
        ('negative_text_global', 'ID-only MCM', 'NegLabel'),
        ('negative_text_fivecrop', 'ID-only + FiveCrop', 'Ours'),
    ]
    for label, base, changed in comparisons:
        for dataset in (*OOD_DATASETS, 'Average'):
            a = all_metrics[(base, dataset)]
            b = all_metrics[(changed, dataset)]
            deltas.append({'comparison': label, 'dataset': dataset,
                           'base': base, 'changed': changed,
                           'delta_auroc_pp': b['auroc']-a['auroc'],
                           'delta_fpr95_pp': b['fpr95']-a['fpr95']})
    write_csv(args.output / 'table2_deltas.csv', deltas,
              ['comparison', 'dataset', 'base', 'changed', 'delta_auroc_pp', 'delta_fpr95_pp'])
    report = [
        'Table 2: Text x View comparison (all numbers in %, deltas in percentage points)',
        'Important: MCM and NegLabel use different score functions; deltas describe performance, not causal interaction.',
        'Existing NegLabel/Ours metrics are imported from reference CSV, not recomputed here.',
        'For complete precision, replace the rounded handoff reference with original complete logs.',
        '',
    ]
    for name in (*OOD_DATASETS, 'Average'):
        report.append(f'[{name}]')
        for method, text, vision, _ in METHODS:
            row = all_metrics[(method, name)]
            report.append(f'  {method:<23} AUROC={row["auroc"]:6.2f}  FPR95={row["fpr95"]:6.2f}  [{row["source"]}]')
        report.append('')
    for label, base, changed in comparisons:
        d = next(x for x in deltas if x['comparison'] == label and x['dataset'] == 'Average')
        report.append(f'{base} -> {changed}: delta AUROC={d["delta_auroc_pp"]:+.3f} pp, '
                      f'delta FPR95={d["delta_fpr95_pp"]:+.3f} pp')
    (args.output / 'table2_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))
    print(f'\n[OUT] {output}\n[OUT] {args.output / "table2_paper.tex"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
