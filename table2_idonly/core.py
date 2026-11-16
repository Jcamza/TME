"""Shared, testable computations for Table 2.

Methods:
- ID-only, global: MCM(100 * normalized_global @ normalized_positive_text.T)
- ID-only, FiveCrop: MCM(100 * h_alpha @ positive_text.T)

The global branch uses Resize(224), CenterCrop(224).
The five-crop branch uses Resize(256), FiveCrop(224), no flips.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
DATASETS = ('ID', 'iNaturalist', 'SUN', 'Places', 'Textures')
OOD_DATASETS = DATASETS[1:]
EXPECTED_COUNTS = {'ID': 50000, 'iNaturalist': 10000, 'SUN': 10000,
                   'Places': 10000, 'Textures': 5640}
DEFAULT_ROOTS = {
    'ID': '/data/Public/imagenet/val',
    'iNaturalist': '/data/Public/NegLabel_MOS/extracted/iNaturalist/iNaturalist/images',
    'SUN': '/data/Public/NegLabel_MOS/extracted/SUN/SUN/images',
    'Places': '/data/Public/NegLabel_MOS/extracted/Places/Places/images',
    'Textures': '/data/Public/DTD/images',
}


def l2_normalize(x: Tensor) -> Tensor:
    # float32 normalization is deterministic and numerically stable across fp16 CLIP devices.
    return F.normalize(x.float(), p=2, dim=-1)


def feature_fusion(global_features: Tensor, crop_features: Tensor, alpha: float) -> Tensor:
    """F.normalize((1-a)*h0+a*mean(h1,...,h5)); no mean renormalization."""
    if not (0 <= alpha <= 1):
        raise ValueError(f'alpha must be in [0,1], got {alpha}')
    if crop_features.ndim != 3 or crop_features.shape[1] != 5:
        raise ValueError('crop_features must have shape [batch,5,dim]')
    if global_features.shape != crop_features.shape[:1] + crop_features.shape[2:]:
        raise ValueError('global/crops have incompatible shapes')
    h0 = l2_normalize(global_features)
    hc = l2_normalize(crop_features).mean(dim=1)
    return l2_normalize((1.0 - alpha) * h0 + alpha * hc)


def mcm_from_features(features: Tensor, positive_text: Tensor, logit_scale: float = 100.0) -> Tensor:
    """Maximum Concept Matching softmax confidence, higher = more ID-like."""
    logits = float(logit_scale) * (l2_normalize(features) @ l2_normalize(positive_text).T)
    return torch.softmax(logits.float(), dim=-1).amax(dim=-1)


def make_transforms():
    """Two *separate* deterministic transforms; same interpolation and CLIP norm."""
    norm = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    after_crop = T.Compose([T.ToTensor(), norm])
    global_transform = T.Compose([
        T.Resize(224, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        after_crop,
    ])
    fivecrop_transform = T.Compose([
        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
        T.FiveCrop(224),
        T.Lambda(lambda crops: torch.stack([after_crop(v) for v in crops], dim=0)),
    ])
    return global_transform, fivecrop_transform


def scan_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {root}')
    # Follow individual image symlinks, not directory symlinks, to avoid cycles.
    paths = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                   key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise RuntimeError(f'No images found under {root}')
    return paths


class PairedViews(torch.utils.data.Dataset):
    def __init__(self, root: Path, paths: Sequence[Path]):
        self.root = root
        self.paths = list(paths)
        self.global_transform, self.fivecrop_transform = make_transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        with Image.open(p) as im:
            img = im.convert('RGB')
            global_image = self.global_transform(img)
            crop_images = self.fivecrop_transform(img)
        return p.relative_to(self.root).as_posix(), global_image, crop_images
