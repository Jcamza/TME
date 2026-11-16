import numpy as np
import pytest
import torch
import torch.nn.functional as F
from neglab_crop.scoring import feature_fusion, neglabel_score_reference, paired_metrics


def official_grouping_algorithm(pos, neg, ngroup=5, t=1.0):
    # Faithful arithmetic of official clip_ood.py:24-46, executed on CPU for this test.
    drop = neg.shape[1] % ngroup
    if drop:
        neg = neg[:, :-drop]
    torch.manual_seed(0)
    idx = torch.randperm(neg.shape[1])
    neg = neg.T
    groups = neg[idx].T.reshape(pos.shape[0], ngroup, -1).contiguous()
    scores = []
    for i in range(ngroup):
        full = torch.cat([pos, groups[:, i, :]], dim=-1) / t
        scores.append(full.softmax(dim=-1)[:, :pos.shape[1]].sum(dim=-1).unsqueeze(-1))
    return torch.cat(scores, dim=-1).mean(dim=-1)


def test_score_matches_official_loop():
    torch.manual_seed(17)
    pos = torch.randn(7, 13) * 8
    neg = torch.randn(7, 107) * 8
    for t in [1., 0.5, 2.]:
        expected = official_grouping_algorithm(pos, neg, ngroup=5, t=t)
        actual = neglabel_score_reference(pos, neg, ngroup=5, t=t)
        assert torch.allclose(actual, expected, atol=1.e-6, rtol=0)


def test_alpha_zero_identity():
    torch.manual_seed(12)
    h0 = F.normalize(torch.randn(4, 12), dim=-1)
    hc = F.normalize(torch.randn(4, 5, 12), dim=-1)
    fused = feature_fusion(h0, hc, 0.)
    assert torch.allclose(fused, h0, atol=2.e-7)
    assert torch.allclose(feature_fusion(h0, hc, 1.).norm(dim=-1), torch.ones(4), atol=1e-6)


def test_metrics_perfect_detection():
    stats = paired_metrics([0.9,0.8,0.7,0.6], [0.1,0.2,0.3,0.4])
    assert stats == {'AUROC': 100.0, 'FPR95': 0.0}


def test_metrics_reverse_detection():
    stats = paired_metrics([0.1,0.2,0.3], [0.7,0.8,0.9])
    assert stats['AUROC'] == 0.0
    assert stats['FPR95'] == 100.0
