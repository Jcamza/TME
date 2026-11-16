"""Read ImageNet-1k and MOS OOD files from unchanged official NegLabel config."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import csv
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int = -1


def relative_to_official(root: Path, path: str) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_dataset_from_config(root: Path, spec, *, recursive=False, limit=0) -> List[Sample]:
    kind = spec['type']
    base = relative_to_official(root, spec['path'])
    if not base.is_dir():
        raise FileNotFoundError(f'Dataset image directory missing: {base}')
    found: List[Sample] = []
    if kind == 'TxtDataset':
        annotation = relative_to_official(root, spec['data_ann'])
        if not annotation.is_file():
            raise FileNotFoundError(f'Annotation missing: {annotation}')
        for line in annotation.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            tokens = line.strip().rsplit(' ', 1)
            name = tokens[0]
            label = int(tokens[1]) if len(tokens) == 2 and tokens[1].lstrip('-').isdigit() else -1
            found.append(Sample(base / name, label))
    elif kind == 'FolderDataset':
        iterator = base.rglob('*') if recursive else base.iterdir()
        found = [Sample(p) for p in sorted(iterator)
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    elif kind == 'JsonDataset':
        annotation = relative_to_official(root, spec['data_ann'])
        record = json.loads(annotation.read_text(encoding='utf-8'))
        file_lookup = {rec['id']: rec['file_name'] for rec in record['images']}
        found = [Sample(base / file_lookup[rec['image_id']]) for rec in record['annotations']]
    elif kind == 'CsvDataset':
        metadata = base / 'metadata.csv'
        with metadata.open(newline='', encoding='utf-8') as f:
            found = [Sample(base / rec['img_filename'], int(rec.get('y', -1)))
                     for rec in csv.DictReader(f) if int(rec['split']) == 2]
    else:
        raise ValueError(f'Unsupported official dataset type {kind}: {spec}')
    if limit > 0:
        found = found[:limit]
    if not found:
        raise ValueError(f'No images found in {base} ({kind})')
    missing = [str(x.path) for x in found if not x.path.is_file()]
    if missing:
        raise FileNotFoundError('Image paths missing (first five):\n' + '\n'.join(missing[:5]))
    return found


class ViewsDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        try:
            with Image.open(s.path) as img:
                views = self.transform(img.convert('RGB'))
        except Exception as e:
            raise RuntimeError(f'Unable to load {s.path}: {e}') from e
        return views, s.label, str(s.path)
