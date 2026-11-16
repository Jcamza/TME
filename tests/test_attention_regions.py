"""CPU smoke tests for new non-training modules; no downloaded weights needed."""
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import torch
import torch.nn.functional as F
from PIL import Image

from neglab_crop.attention_regions import (
    select_attention_mass, masked_clip_image, boxes_from_mask,
    box_crops_as_clip, _get_cls_attn_from_mha, _ncut_mask, DINOv2ObjectRegions,
)
from neglab_crop.attention_runner import RegionDataset, RegionCache
from neglab_crop.data import Sample
from neglab_crop.preprocess import ImageViews


def test_mass_is_cumulative_not_70_percent_of_token_count():
    a = torch.tensor([[[.4,.3],[.2,.1]]],dtype=torch.float)
    keep = select_attention_mass(a, .7)
    assert int(keep.sum()) == 2
    assert keep[0,0,0] and keep[0,0,1]
    assert not keep[0,1,0]


def test_mask_and_bbox_preserve_shape():
    keep = torch.zeros(1,14,14,dtype=torch.bool)
    keep[:,3:10,4:11] = True
    x = torch.ones(1,3,224,224)
    out = masked_clip_image(x,keep)
    assert out.shape == x.shape
    assert out[0,:,0,0].sum() == 0
    assert out[0,:,70,80].sum() == 3
    b = boxes_from_mask(keep, size=224)
    crop = box_crops_as_clip(torch.rand(1,3,224,224),b)
    assert crop.shape == (1,3,224,224)
    assert 0 <= b[0][0] < b[0][2] <= 224


def test_clip_cls_matches_actual_torch_mha_weights():
    torch.manual_seed(11)
    mha = torch.nn.MultiheadAttention(embed_dim=8,num_heads=2)
    mha.eval()
    x = torch.randn(5,2,8)
    inferred = _get_cls_attn_from_mha(mha,x)
    _, attn = mha(x,x,x,need_weights=True,average_attn_weights=False)
    target = attn[:,:,:1,1:].mean(dim=1).squeeze(1)
    target /= target.sum(-1,keepdim=True)
    assert inferred.shape == (2,2,2)
    torch.testing.assert_close(inferred.flatten(1),target,rtol=1e-5,atol=1e-6)


def test_region_dataset_global_is_identical_to_old_preprocess(tmp_path):
    import numpy as np
    arr = (np.random.default_rng(42).random((240,320,3))*255).astype('uint8')
    path = tmp_path/'a.png'
    Image.fromarray(arr).save(path)
    data=RegionDataset([Sample(path)])[0]
    with Image.open(path) as im:
        original = ImageViews()(im)[0]
    torch.testing.assert_close(data[0],original)
    assert data[1].shape==(3,224,224)


def test_ncut_safe_fallback_for_degenerate_graph():
    aff=torch.ones(16,16)
    sal=torch.linspace(0,1,16)
    mask, reason=_ncut_mask(aff,sal,4,4)
    assert mask.shape==(4,4) and 0<mask.sum()<=16
    assert isinstance(reason,str)


def test_region_cache_persists(tmp_path):
    sample=tmp_path/'image.jpg'
    sample.write_bytes(b'abc')
    path=tmp_path/'c.sqlite'
    cache=RegionCache(path)
    key=cache.key('cls',sample,{'weight':.7})
    assert cache.get(key) is None
    cache.set(key,{'box':[1,2,3,4]});cache.commit();cache.close()
    cache=RegionCache(path)
    assert cache.get(key)=={'box':[1,2,3,4]}
    cache.close()


class TinyAttn(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query=torch.nn.Linear(dim,dim)
        self.key=torch.nn.Linear(dim,dim)
        self.value=torch.nn.Linear(dim,dim)
    def forward(self, x):
        return x + .01*self.value(x)


class TinyLayer(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.attention=types.SimpleNamespace() # replace with nn.Module nested
        self.attention=torch.nn.Module()
        self.attention.add_module('attention',TinyAttn(dim))
    def forward(self,x):
        return self.attention.attention(x)


class TinyDinov2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(patch_size=14,num_attention_heads=2)
        self.project=torch.nn.Linear(3,8)
        self.encoder=torch.nn.Module()
        self.encoder.layer=torch.nn.ModuleList([TinyLayer(8) for _ in range(3)])
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        return cls()
    def forward(self,pixel_values):
        x=F.avg_pool2d(pixel_values,kernel_size=14).flatten(2).transpose(1,2)
        x=self.project(x)
        x=torch.cat([x.mean(dim=1,keepdim=True),x],dim=1)
        for layer in self.encoder.layer:
            x=layer(x)
        return SimpleNamespace(last_hidden_state=x)


def test_object_dino_guided_fake_forward_and_box(monkeypatch):
    torch.manual_seed(1)
    fake=types.ModuleType('transformers')
    fake.Dinov2Model=TinyDinov2
    monkeypatch.setitem(sys.modules,'transformers',fake)
    dino=DINOv2ObjectRegions('fake',device='cpu',clusters=3,graph_neighbors=2,n_init=1)
    regs=dino.boxes(torch.rand(2,3,28,28))
    dino.close()
    assert len(regs)==2
    for region in regs:
        x0,y0,x1,y1=region.box
        assert 0<=x0<x1<=28 and 0<=y0<y1<=28
        assert region.selected_heads>0
        assert 0<region.box_fraction<=1
