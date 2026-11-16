"""Feature fusion and equivalence-tested vectorization of official NegLabel grouping.

Official source: https://github.com/tmlr-group/NegLabel/blob/main/mmcls/models/ood/clip_ood.py
"""
import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def feature_fusion(global_feature: torch.Tensor, crop_features: torch.Tensor,
                   alpha: float) -> torch.Tensor:
    """Inputs BxD and BxRxD are unit length; output BxD, unit length."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha outside [0,1]')
    if crop_features.ndim != 3 or global_feature.shape != crop_features.shape[::2]:
        raise ValueError('Expected global=(B,D), crops=(B,R,D)')
    if alpha == 0.0:
        return global_feature  # strict equivalence to original; no second normalization
    local = crop_features.mean(dim=1)
    return l2_normalize((1.0 - alpha) * global_feature + alpha * local)


def neglabel_score_reference(pos: torch.Tensor, neg: torch.Tensor, *, t: float,
                             ngroup: int, perm: torch.Tensor = None) -> torch.Tensor:
    """Mathematics of official grouping(), used only for tests/diagnostics.

    The production runner calls original ScalableClassifier.grouping() directly.
    Note: official code truncates trailing negatives, then uses a fixed seed-0 permutation.
    """
    if ngroup <= 1:
        raise ValueError('Reference helper requires ngroup > 1 (same as production).')
    n = (neg.shape[1] // ngroup) * ngroup
    if n == 0:
        raise ValueError('Fewer negative labels than groups')
    neg = neg[:, :n]
    if perm is None:
        gen = torch.Generator(device='cpu').manual_seed(0)
        perm = torch.randperm(n, generator=gen).to(neg.device)
    groups = neg[:, perm].reshape(pos.shape[0], ngroup, -1)
    a = torch.logsumexp(pos / t, dim=-1, keepdim=True)
    b = torch.logsumexp(groups / t, dim=-1)
    return torch.sigmoid(a - b).mean(dim=1)


def paired_metrics(id_scores, ood_scores):
    """AUROC and FPR at >=95% ID acceptance; returns percentages.

    ROC-based interpolation follows usual OOD evaluation protocols. `fpr95` is
    computed via first ROC threshold that reaches TPR>=0.95 (discrete ROC).
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    ins = np.asarray(id_scores, dtype=np.float64)
    outs = np.asarray(ood_scores, dtype=np.float64)
    if not len(ins) or not len(outs):
        raise ValueError('ID and OOD sets must both be nonempty')
    y = np.concatenate([np.ones(len(ins), dtype=np.int8), np.zeros(len(outs), dtype=np.int8)])
    s = np.concatenate([ins, outs])
    auc = float(roc_auc_score(y, s) * 100.)
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[idx[0]] * 100.) if len(idx) else 100.0
    return {'AUROC': auc, 'FPR95': fpr95}
