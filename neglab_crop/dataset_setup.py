"""Discover/prepare the exact MOS four-OOD benchmark for NegLabel.

Python standard-library only. Never modifies source images or NegLabel model code.
The image directories created under official_root/data/ood_data are flat symlink
views, matching official custom_config/clip.py and the paired runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DATASET_NAMES = ('iNaturalist', 'SUN', 'Places', 'Textures')
EXPECTED = {'iNaturalist': 10000, 'SUN': 10000, 'Places': 10000, 'Textures': 5640}
ID_EXPECTED = 50000
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# MOS paper / official MOS GitHub README (curated, *not* raw full datasets):
MOS_URL = 'https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/{name}.tar.gz'
DTD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'


def official_image_dir(official_root: Path, name: str) -> Path:
    if name == 'Textures':
        return official_root / 'data' / 'ood_data' / 'Textures' / 'dtd' / 'images_collate'
    return official_root / 'data' / 'ood_data' / name / 'images'


def count_images(directory: Path, recursive: bool = True, cap: Optional[int] = None) -> int:
    if not directory.is_dir():
        return 0
    n = 0
    # This follows directory symlinks at the top level, but not nested directory links.
    if not recursive:
        return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        n += sum(1 for fn in files if Path(fn).suffix.lower() in IMAGE_EXTS
                 and (Path(root) / fn).is_file())
        if cap is not None and n > cap:
            return n
    return n


def sample_images(directory: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for fn in sorted(files):
            if Path(fn).suffix.lower() in IMAGE_EXTS and (Path(root) / fn).is_file():
                yield Path(root) / fn


def named_candidate_kind(dirname: str) -> Optional[str]:
    s = dirname.casefold().replace('-', '_')
    # Avoid treating SUN397 or Places365 as the curated MOS subsets.
    if s in ('inaturalist', 'inaturalist_ood', 'mos_inaturalist'):
        return 'iNaturalist'
    if s in ('sun', 'sun_ood', 'mos_sun'):
        return 'SUN'
    if s in ('places', 'places_ood', 'mos_places'):
        return 'Places'
    if s in ('textures', 'texture', 'dtd'):
        return 'Textures'
    return None


def normalized_name(s: str) -> str:
    for name in DATASET_NAMES:
        if s.casefold() == name.casefold():
            return name
    raise ValueError(f'Unknown dataset {s!r}. Options: {", ".join(DATASET_NAMES)}')


def image_root_options(root: Path, name: str) -> List[Path]:
    guesses = [root, root / 'images', root / 'images_collate', root / 'dtd' / 'images',
               root / 'dtd' / 'images_collate', root / name / 'images',
               root / name / 'dtd' / 'images', root / name / 'dtd' / 'images_collate']
    out, seen = [], set()
    for p in guesses:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def best_image_root(root: Path, name: str) -> Tuple[Optional[Path], int]:
    expected = EXPECTED[name]
    best, max_count = None, 0
    for p in image_root_options(root, name):
        if not p.is_dir():
            continue
        num = count_images(p, recursive=True, cap=expected + 1)
        if num == expected:
            return p.resolve(), num
        if num > max_count:
            best, max_count = p.resolve(), num
    return best, max_count


def candidate_dirs(search_root: Path, max_depth: int = 4) -> Iterable[Tuple[str, Path]]:
    """Bounded folder-name discovery; does not traverse individual image files."""
    if not search_root.is_dir():
        return
    # Very large uninteresting trees are skipped, but source folder names are matched first.
    prune = {'train', 'val', 'validation', 'test', 'checkpoints', 'models', '.git',
             '__pycache__', 'node_modules', 'split_fewshot', 'venv', 'envs',
             'site-packages', 'features', 'logs', 'results', 'images',
             'images_collate'}
    base_depth = len(search_root.parts)
    for root_str, dirs, _ in os.walk(search_root, followlinks=False):
        root = Path(root_str)
        rel_depth = len(root.parts) - base_depth
        if rel_depth >= max_depth:
            dirs[:] = []
            continue
        keep = []
        for dirname in sorted(dirs):
            subdir = root / dirname
            name = named_candidate_kind(dirname)
            if name is not None:
                yield name, subdir
                continue
            if dirname.casefold() in prune or dirname.startswith('.'):
                continue
            # Never walk into shared 1000-class ImageNet roots.
            if re.fullmatch(r'n\d{8}', dirname):
                continue
            keep.append(dirname)
        dirs[:] = keep


def parse_sources(values: List[str]) -> Dict[str, Path]:
    results = {}
    for token in values:
        if '=' not in token:
            raise ValueError(f'Expected NAME=PATH for --source, got {token!r}')
        name, path = token.split('=', 1)
        results[normalized_name(name.strip())] = Path(path).expanduser().resolve()
    return results


def discover(name: str, *, official_root: Path, project_root: Path,
             search_roots: List[Path], explicit: Dict[str, Path]) -> Tuple[Optional[Path], int, str]:
    options: List[Tuple[str, Path]] = []
    if name in explicit:
        options.append(('explicit', explicit[name]))
    env_name = {'iNaturalist': 'OOD_INATURALIST', 'SUN': 'OOD_SUN',
                'Places': 'OOD_PLACES', 'Textures': 'OOD_TEXTURES'}[name]
    if os.environ.get(env_name):
        options.append(('config.env', Path(os.environ[env_name]).expanduser()))
    if name == 'Textures':
        options.extend([('known DTD', Path('/data/Public/DTD/images')),
                        ('known DTD', Path('/data/Public/dtd/images'))])
    for base in [Path('/data/Public/ood_data'), Path('/data/Public/ImageNet_OOD_dataset'),
                 Path('/data/Public/imagenet_ood_dataset'), official_root / 'data' / 'ood_data',
                 project_root / 'data' / 'ood_data']:
        options.extend([('known OOD layout', base / name),
                        ('known OOD layout', base / ('dtd' if name == 'Textures' else name))])
    for base in search_roots:
        for ds_name, candidate in candidate_dirs(base):
            if ds_name == name:
                options.append((f'found under {base}', candidate))
    seen = set()
    best_bad = (None, 0, '')
    for label, candidate in options:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        root, count = best_image_root(candidate, name)
        if count == EXPECTED[name]:
            return root, count, label
        if count > best_bad[1]:
            best_bad = (root, count, f'{label} (count mismatch)')
    return best_bad


def _check_id(project_root: Path, image_dir: Path) -> dict:
    count = count_images(image_dir, cap=ID_EXPECTED + 1)
    cats = len([p for p in image_dir.iterdir() if p.is_dir()]) if image_dir.is_dir() else 0
    ann = project_root / 'data' / 'imagenet_val_ood_index.txt'
    ann_count = sum(1 for _ in ann.open(encoding='utf-8')) if ann.is_file() else 0
    return {'path': str(image_dir), 'count': count, 'categories': cats,
            'expected': ID_EXPECTED, 'annotation': str(ann), 'annotation_rows': ann_count,
            'ready': count == ID_EXPECTED and cats == 1000 and ann_count == ID_EXPECTED}


def create_id_annotation(project_root: Path, id_dir: Path):
    id_dir = id_dir.resolve()
    n = count_images(id_dir, cap=ID_EXPECTED + 1)
    cats = sorted(p for p in id_dir.iterdir() if p.is_dir()) if id_dir.is_dir() else []
    if n != ID_EXPECTED or len(cats) != 1000:
        raise RuntimeError(f'ID directory not canonical ImageNet-val: {id_dir} ({n} images, {len(cats)} classes)')
    out = project_root / 'data' / 'imagenet_val_ood_index.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if sum(1 for _ in out.open(encoding='utf-8')) != ID_EXPECTED:
            raise RuntimeError(f'Existing annotation has wrong row count: {out}; please inspect manually')
        print(f'[ID] Reusing annotation: {out}', flush=True)
        return out
    rows = [p.relative_to(id_dir).as_posix() for p in sample_images(id_dir)]
    if len(rows) != ID_EXPECTED:
        raise RuntimeError('Unexpected ImageNet image count changed during indexing')
    # -1 is a sentinel label: paired OOD AUROC/FPR95 do not require classification labels.
    # Never report this index as a valid classification accuracy annotation.
    tmp = out.with_suffix('.txt.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for image in rows:
            f.write(f'{image} -1\n')
    tmp.replace(out)
    print(f'[ID] Generated {len(rows)} entries: {out}', flush=True)
    return out


def flat_output_name(path: Path, src: Path) -> str:
    rel = path.relative_to(src)
    name = '__'.join(rel.parts)
    if len(name.encode('utf-8')) < 240:
        return name
    import hashlib
    suffix = hashlib.sha256(str(rel).encode('utf-8')).hexdigest()[:14]
    return name[:180] + '__' + suffix + path.suffix.lower()


def prepare_flat_view(src: Path, dest: Path, name: str) -> int:
    src = src.resolve()
    expected = EXPECTED[name]
    got = count_images(src, cap=expected + 1)
    if got != expected:
        raise RuntimeError(f'{name}: source count {got}, expected {expected}: {src}')
    if dest.exists() or dest.is_symlink():
        present = count_images(dest, recursive=False, cap=expected + 1)
        if present == expected:
            print(f'[REUSE] {name}: existing prepared flat directory with {present} images: {dest}')
            return present
        raise RuntimeError(f'{name}: refusing to overwrite existing prepared path {dest}; has {present} flat images')
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + dest.name + '_stage_', dir=dest.parent))
    try:
        used = set()
        n = 0
        for image in sample_images(src):
            target_name = flat_output_name(image, src)
            if target_name in used:
                raise RuntimeError(f'Collision creating flattened names: {target_name}')
            used.add(target_name)
            (staging / target_name).symlink_to(image.resolve())
            n += 1
        if n != expected:
            raise RuntimeError(f'{name}: found {n} images, expected {expected}')
        staging.rename(dest)
        print(f'[LINK] {name}: {n} symlinks -> {dest} (source {src})', flush=True)
        return n
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _archive_name(name: str) -> str:
    return 'dtd-r1.0.1.tar.gz' if name == 'Textures' else f'{name}.tar.gz'


def _download_archive(name: str, download_root: Path) -> Path:
    download_root.mkdir(parents=True, exist_ok=True)
    archive = download_root / _archive_name(name)
    if archive.is_file() and tarfile.is_tarfile(archive):
        print(f'[DOWNLOAD] Reusing existing tar archive: {archive}')
        return archive
    if not shutil.which('curl'):
        raise RuntimeError('curl is required for resumable archive downloads')
    urls = [DTD_URL] if name == 'Textures' else [MOS_URL.format(name=name),
                                                   MOS_URL.format(name=name).replace('https://', 'http://', 1)]
    part = archive.with_suffix(archive.suffix + '.part')
    for url in urls:
        print(f'[DOWNLOAD] {name}: {url}', flush=True)
        cmd = ['curl', '--fail', '--location', '--retry', '3', '--retry-delay', '3',
               '--continue-at', '-', '--output', str(part), url]
        completed = subprocess.run(cmd, check=False)
        if completed.returncode == 0 and tarfile.is_tarfile(part):
            part.replace(archive)
            print(f'[DOWNLOAD] Saved {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)')
            return archive
        print(f'[WARN] Download failed / invalid tarball from {url}: exit={completed.returncode}')
        # Discard the partial file before trying a different URL.
        if part.exists():
            part.unlink()
    raise RuntimeError(f'Cannot download valid {name} archive. See official MOS dataset links.')


def _safe_extract(archive: Path, destination: Path):
    """Extract only regular files/dirs; reject path traversal and archive links."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, mode='r:gz') as tf:
        for member in tf:
            p = Path(member.name)
            if p.is_absolute() or '..' in p.parts:
                raise RuntimeError(f'Unsafe archive entry: {member.name}')
            target = (destination / p).resolve()
            if not target.is_relative_to(base):
                raise RuntimeError(f'Unsafe archive target: {member.name}')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is None:
                    raise RuntimeError(f'Cannot read tar entry: {member.name}')
                with source, target.open('wb') as dest_file:
                    shutil.copyfileobj(source, dest_file)
            else:
                raise RuntimeError(f'Unsupported archive special entry (symlink/device): {member.name}')


def _download_and_find(name: str, project_root: Path) -> Tuple[Path, str]:
    dl_root = project_root / 'data' / 'mos_downloads'
    archive = _download_archive(name, dl_root)
    unpack = dl_root / 'extracted' / name
    root, count = best_image_root(unpack, name)
    if count != EXPECTED[name]:
        print(f'[EXTRACT] {archive} -> {unpack}', flush=True)
        _safe_extract(archive, unpack)
        root, count = best_image_root(unpack, name)
    if count != EXPECTED[name]:
        # MOS archives sometimes have an extra wrapper folder.
        matches = [(r, n) for ds, cand in candidate_dirs(unpack, max_depth=4) if ds == name
                   for r, n in [best_image_root(cand, name)]]
        matches.extend((r, n) for guessed in unpack.iterdir() if guessed.is_dir()
                       for r, n in [best_image_root(guessed, name)])
        for r, n in matches:
            if n == EXPECTED[name]:
                root, count = r, n
                break
    if count != EXPECTED[name] or root is None:
        raise RuntimeError(f'Extracted {name} images count {count}, expected {EXPECTED[name]}; inspect {unpack}')
    return root, str(archive)


def _write_report(project_root: Path, action: str, records: dict):
    out_dir = project_root / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'action': action, **records}
    (out_dir / 'mos_dataset_status.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out_dir / 'mos_dataset_status.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'status', 'count', 'expected', 'path', 'origin'])
        for name, d in report.get('ood', {}).items():
            writer.writerow([name, d.get('status'), d.get('count'), EXPECTED[name],
                             d.get('path'), d.get('origin')])
    print(f'[REPORT] {out_dir / "mos_dataset_status.json"}')


def run(args) -> int:
    project_root = args.project_root.expanduser().resolve()
    official_root = args.official_root.expanduser().resolve()
    id_path = args.id_path.expanduser().resolve()
    search_roots = [p.expanduser().resolve() for p in args.search_root]
    explicit = parse_sources(args.source)
    if not project_root.is_dir():
        raise RuntimeError(f'Project root is missing: {project_root}')
    if args.action in ('prepare', 'verify') and not (official_root / 'custom_config' / 'clip.py').is_file():
        raise RuntimeError(f'Official NegLabel checkout missing: {official_root}')
    if args.action == 'prepare':
        create_id_annotation(project_root, id_path)

    id_status = _check_id(project_root, id_path)
    print(f'[ID] ImageNet-val {id_status["count"]}/{ID_EXPECTED} images, '
          f'{id_status["categories"]}/1000 folders, annotation {id_status["annotation_rows"]}/{ID_EXPECTED}')
    records = {'id': id_status, 'ood': {}}
    failures = []
    for name in DATASET_NAMES:
        prepared = official_image_dir(official_root, name)
        if args.action == 'verify':
            image_count = count_images(prepared, recursive=False, cap=EXPECTED[name] + 1)
            root, count, origin = prepared, image_count, 'official prepared flat path'
        else:
            root, count, origin = discover(name, official_root=official_root,
                                           project_root=project_root,
                                           search_roots=search_roots, explicit=explicit)
            if args.action == 'prepare':
                if count != EXPECTED[name] and args.download_missing:
                    root, origin = _download_and_find(name, project_root)
                    count = EXPECTED[name]
                if count == EXPECTED[name] and root is not None:
                    prepare_flat_view(root, prepared, name)
                    root = prepared
                    count = count_images(prepared, recursive=False, cap=EXPECTED[name] + 1)
        status = 'READY' if count == EXPECTED[name] else ('MISSING' if count == 0 else 'SIZE_MISMATCH')
        records['ood'][name] = {'status': status, 'path': str(root) if root else '',
                                'count': count, 'expected': EXPECTED[name], 'origin': origin}
        print(f'[{status}] {name:12} {count:5}/{EXPECTED[name]:5}  {root or "(not found)"} ({origin})', flush=True)
        if status != 'READY':
            failures.append(name)
    _write_report(project_root, args.action, records)
    if not id_status['ready']:
        print('[ERROR] ID dataset not ready: check ImageNet validation folder / ID index', file=sys.stderr)
        failures.insert(0, 'ImageNet-val')
    if failures:
        print('[INCOMPLETE] ' + ', '.join(failures) + '. '
              + ('Run scripts/06_prepare_mos_datasets.sh --download-missing' if args.action == 'check'
                 else 'Prepare exact MOS versions before full evaluation.'), flush=True)
        return 2 if args.action != 'check' or args.strict else 0
    print('[READY] Four-OOD dataset check passed. Counts alone do not prove MOS provenance.', flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Discover, download or prepare MOS ImageNet-1k four-OOD benchmark')
    p.add_argument('action', choices=['check', 'prepare', 'verify'])
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    p.add_argument('--official-root', type=Path, default=Path('/data/hx/code/NegLabel'))
    p.add_argument('--id-path', type=Path, default=Path('/data/Public/imagenet/val'))
    p.add_argument('--search-root', type=Path, action='append', default=[],
                   help='Folder to scan for dataset names (up to 4 directory levels). Repeatable.')
    p.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                   help='Explicit local dataset directory; can repeat. Must contain exactly MOS expected count.')
    p.add_argument('--download-missing', action='store_true',
                   help='Explicitly download missing official MOS archives / DTD from original website')
    p.add_argument('--strict', action='store_true', help='Exit nonzero on missing data even in check mode')
    args = p.parse_args(argv)
    if not args.search_root:
        args.search_root = [Path('/data/Public'), args.official_root / 'data' / 'ood_data',
                            args.project_root / 'data' / 'mos_downloads' / 'extracted']
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
