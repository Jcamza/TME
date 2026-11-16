"""Exercise new runner end-to-end on CPU with a fake (not pretrained) CLIP.
The fake model is only used in tests; production never allows random weights.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from neglab_crop import attention_alpha_runner as attention_runner
from neglab_crop.scoring import neglabel_score_reference


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,8,16,16)
        self.transformer=nn.Module()
        self.transformer.resblocks=nn.ModuleList([nn.Module()])
        self.transformer.resblocks[0].attn=nn.MultiheadAttention(8,2)
        self.proj=nn.Linear(8,6)
    def forward(self, x):
        x=self.conv(x).flatten(2).permute(2,0,1)
        cls=x.mean(0,keepdim=True)
        x=torch.cat([cls,x],dim=0)
        x,_=self.transformer.resblocks[0].attn(x,x,x,need_weights=False)
        return self.proj(x[0])


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=TinyViT()
    def encode_image(self,x):
        return self.visual(x)


class DummyOfficial:
    def __init__(self):
        self.classifier=SimpleNamespace(
            model=TinyCLIP(),
            text_features_pos=F.normalize(torch.randn(5,6),dim=1),
            text_features_neg=F.normalize(torch.randn(20,6),dim=1))
        self.ngroup=5
        self.group_fuse_num=None
        self.t=1
    def grouping(self,p,n,num,ngroup,random_permute):
        return neglabel_score_reference(p,n,t=1,ngroup=ngroup)
    def forward(self,img):
        from neglab_crop.runner import _normalize_like_official,_official_score
        return _official_score(self,_normalize_like_official(self.classifier.model.encode_image(img)))


def test_attention_runner_partial_cpu(monkeypatch,tmp_path):
    torch.manual_seed(1)
    idroot=tmp_path/'id'
    oodroot=tmp_path/'ood'
    idroot.mkdir();oodroot.mkdir()
    for i in range(4):
        arr=np.random.default_rng(i).integers(0,256,(240,300,3),dtype='uint8')
        Image.fromarray(arr).save(idroot/f'{i}.png')
        Image.fromarray(255-arr).save(oodroot/f'{i}.png')
    ann=tmp_path/'id.txt'
    ann.write_text(''.join(f'{i}.png -1\n' for i in range(4)))
    cfg=SimpleNamespace(data=SimpleNamespace(
        id_data={'type':'TxtDataset','path':str(idroot),'data_ann':str(ann)},
        ood_data=[{'name':'Textures','type':'FolderDataset','path':str(oodroot)}]))
    fake=DummyOfficial()
    monkeypatch.setattr(attention_runner,'load_official',lambda *a,**k:(cfg,fake))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    # Minimal stand-in for official evaluate_all; not used for research numbers.
    from sklearn.metrics import roc_auc_score,roc_curve
    def eval_all(id_s,ood_s):
        y=np.r_[np.ones(len(id_s)),np.zeros(len(ood_s))]
        score=np.r_[id_s,ood_s]
        auroc=roc_auc_score(y,score)
        fpr,tpr,_=roc_curve(y,score)
        fpr95=fpr[np.where(tpr>=.95)[0][0]]
        return auroc,auroc,auroc,fpr95
    mmcls=types.ModuleType('mmcls')
    utils=types.ModuleType('mmcls.utils')
    oodeval=types.ModuleType('mmcls.utils.ood_evaluate')
    oodeval.evaluate_all=eval_all
    monkeypatch.setitem(sys.modules,'mmcls',mmcls)
    monkeypatch.setitem(sys.modules,'mmcls.utils',utils)
    monkeypatch.setitem(sys.modules,'mmcls.utils.ood_evaluate',oodeval)
    dest=tmp_path/'result'
    rows=attention_runner.main([
        '--official-root',str(tmp_path), '--output-dir',str(dest),
        '--skip-dino','--include-fivecrop','--check-official-parity',
        '--batch-size','2','--view-batch-size','4','--workers','0',
        '--no-plot','--dataset','Textures','--limit-id','4','--limit-ood','4',
        '--visualize-n','1', '--cache-db',str(tmp_path/'c.sqlite')])
    methods={r['method'] for r in rows}
    assert methods=={'official_original','fivecrop_alpha0.70','cls_attention_mask_mass70_alpha0.70'}
    assert (dest/'scores/ID.csv').is_file()
    assert (dest/'scores/Textures.csv').is_file()
    assert (dest/'diagnostics/Textures.csv').is_file()
    assert (dest/'examples/Textures_0000.jpg').is_file()
    assert (dest/'comparison.csv').is_file()
    # Cover DINO proposal cache, box resize, and scorer wiring as well.
    from neglab_crop.attention_regions import ObjectRegion
    class StubDino:
        def __init__(self,*args,**kwargs):
            pass
        def boxes(self,raw):
            return [ObjectRegion((30,25,190,185),.35,.51,20,'') for _ in raw]
        def close(self):
            pass
    monkeypatch.setattr(attention_runner,'DINOv2ObjectRegions',StubDino)
    dest2=tmp_path/'result_dino'
    dino_rows=attention_runner.main([
        '--official-root',str(tmp_path), '--output-dir',str(dest2),
        '--include-fivecrop','--check-official-parity',
        '--batch-size','2','--view-batch-size','4','--workers','0',
        '--no-plot','--dataset','Textures','--limit-id','4','--limit-ood','4',
        '--visualize-n','1', '--cache-db',str(tmp_path/'dino.sqlite')])
    assert 'object_dino_bbox_alpha0.70' in {r['method'] for r in dino_rows}
    assert (dest2/'diagnostics/Textures_summary.json').is_file()

    # Alpha-grid branch: region extraction must be shared, and alpha=0 must
    # reproduce the baseline score *exactly* on every dataset and image.
    sweep_dest=tmp_path/'sweep'
    sweep_rows=attention_runner.main([
        '--official-root',str(tmp_path),'--output-dir',str(sweep_dest),
        '--alpha-grid','0,0.05,0.7,1.0','--cls-mass','0.95',
        '--check-official-parity','--batch-size','2',
        '--view-batch-size','4','--workers','0','--no-plot',
        '--dataset','Textures','--limit-id','4','--limit-ood','4',
        '--visualize-n','0','--cache-db',str(tmp_path/'dino.sqlite')])
    names={r['method'] for r in sweep_rows}
    assert len(names)==9  # original + 4 CLS + 4 Object-DINO
    for name in ('ID','Textures'):
        import csv
        with (sweep_dest/'scores'/f'{name}.csv').open(newline='') as f:
            score_rows=list(csv.DictReader(f))
        assert len(score_rows)==4
        for row in score_rows:
            assert row['official_original']==row['cls_attention_mask_mass95_alpha0.00']
            assert row['official_original']==row['object_dino_bbox_alpha0.00']
    assert (sweep_dest/'comparison.csv').exists()
