"""Unmodified NegLabel ImageNet global preprocessing plus deterministic 5/10-crop views.

Sources:
- NegLabel official mmcls/datasets/ood_dataset.py (global preprocessing):
  https://github.com/tmlr-group/NegLabel/blob/main/mmcls/datasets/ood_dataset.py
- torchvision.transforms.FiveCrop and TenCrop:
  https://pytorch.org/vision/stable/generated/torchvision.transforms.FiveCrop.html
  https://pytorch.org/vision/stable/generated/torchvision.transforms.TenCrop.html
- Krizhevsky, Sutskever, Hinton, NIPS 2012, Sec. 4.1 (multi-crop test inference).
"""
from typing import List
import torch
from PIL import Image
from torchvision import transforms as T

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class ImageViews:
    def __init__(self, crop_mode='fivecrop', crop_resize=256, crop_size=224):
        if crop_mode not in ('fivecrop', 'tencrop'):
            raise ValueError('crop_mode must be fivecrop or tencrop')
        if crop_resize < crop_size:
            raise ValueError('crop_resize must be >= crop_size')
        self.crop_mode = crop_mode
        # Exact upstream transform (Resize scalar uses shorter edge; default interpolation BILINEAR).
        self.global_transform = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD),
        ])
        self.resize = T.Resize(crop_resize)
        self.crop = T.FiveCrop(crop_size) if crop_mode == 'fivecrop' else T.TenCrop(crop_size)
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original = self.global_transform(img)
        crops = [self.to_tensor(c) for c in self.crop(self.resize(img))]
        if any(t.shape != original.shape for t in crops):
            raise RuntimeError('view dimensions differ; crop_size must be 224 for ViT-B/16')
        return torch.stack([original, *crops], dim=0)

    @property
    def n_views(self) -> int:
        return 6 if self.crop_mode == 'fivecrop' else 11
