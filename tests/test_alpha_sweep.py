from types import SimpleNamespace
from pathlib import Path
import csv
import pytest
import torch
from neglab_crop.attention_alpha_runner import _resolve_alpha_grid
from neglab_crop.alpha_sweep_report import build_summary
from neglab_crop.scoring import feature_fusion


def test_alpha_grid_default_and_values():
    assert _resolve_alpha_grid(SimpleNamespace(alpha_grid='',alpha=0.7))==[0.7]
    assert _resolve_alpha_grid(SimpleNamespace(alpha_grid='0,0.05,0.1,1',alpha=.7))==[0.,.05,.1,1.]


@pytest.mark.parametrize('value',['nan','inf','-0.1','1.1','0.1,0.1','0.101,0.102','oops',''])
def test_alpha_grid_invalid(value):
    if value == '':
        return  # legacy single-alpha mode
    with pytest.raises(ValueError):
        _resolve_alpha_grid(SimpleNamespace(alpha_grid=value,alpha=.7))


def test_alpha_zero_original_exact():
    a=torch.nn.functional.normalize(torch.randn(5,32),dim=1)
    b=torch.nn.functional.normalize(torch.randn(5,1,32),dim=2)
    assert torch.equal(feature_fusion(a,b,0.),a)


def test_alpha_report_schema(tmp_path):
    p=tmp_path/'comparison.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','method','AUROC','FPR95',
             'delta_AUROC_vs_original','delta_FPR95_vs_original'])
        w.writeheader()
        for method in ('official_original','cls_attention_mask_mass95_alpha0.00',
                       'object_dino_bbox_alpha0.20','fivecrop_alpha0.70'):
            w.writerow({'dataset':'SUN','method':method,'AUROC':95,
                        'FPR95':22,'delta_AUROC_vs_original':0,
                        'delta_FPR95_vs_original':0})
    result=build_summary(p)
    assert len(result)==2
    assert result[0]['branch']=='CLS95 mask'
    assert result[0]['alpha']==0.
    assert result[1]['branch']=='Object-DINO bbox'
