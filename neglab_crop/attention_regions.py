"""Training-free CLS-patch and Object-DINO-guided region proposals.

CLIP branch: exact final visual block CLS-to-patch attention, NOT DINO.
DINO branch: Q/K/V patch similarity, across-layer head clustering, then
an adapted kNN normalized-cut graph for a single region box. This is an
Object-DINO-GUIDED ADAPTATION, NOT a bit-identical reproduction of its
official TokenCut object-discovery benchmark.

No model parameters are updated and no OOD labels are used in selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .preprocess import MEAN, STD

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def select_attention_mass(attn: torch.Tensor, mass: float = .7) -> torch.Tensor:
    """Smallest per-image top-k set whose normalized attention mass >= mass.

    Input [B,H,W] (positive, unnormalized allowed); output bool [B,H,W].
    Ties are resolved by torch sort; *not* a threshold on the top 70% of tokens.
    """
    if not 0 < mass <= 1:
        raise ValueError('mass must be in (0,1]')
    if attn.ndim != 3:
        raise ValueError('attention must have [B,H,W] shape')
    b, h, w = attn.shape
    a = torch.nan_to_num(attn.float().flatten(1), nan=0., posinf=0., neginf=0.).clamp_min(0.)
    denom = a.sum(1, keepdim=True)
    a = torch.where(denom > 1e-10, a / denom.clamp_min(1e-10),
                    torch.full_like(a, 1 / (h * w)))
    val, order = a.sort(dim=1, descending=True, stable=True)
    cumulative = val.cumsum(dim=1)
    n = (cumulative < mass).sum(dim=1).clamp(max=h*w-1) + 1
    ranks = torch.arange(h*w, device=a.device).unsqueeze(0).expand(b, -1)
    sorted_keep = ranks < n[:, None]
    out = torch.zeros_like(sorted_keep).scatter_(1, order, sorted_keep)
    return out.view(b, h, w)


def masked_clip_image(global_normalized: torch.Tensor, patch_keep: torch.Tensor) -> torch.Tensor:
    """Keep selected image patch pixels; replace others by CLIP RGB mean (0 normalized)."""
    if global_normalized.ndim != 4 or patch_keep.ndim != 3:
        raise ValueError('Expect global [B,3,H,W] and mask [B,h,w]')
    mask = F.interpolate(patch_keep[:, None].float(),
                         size=global_normalized.shape[-2:], mode='nearest')
    return global_normalized * mask


def boxes_from_mask(mask: torch.Tensor, *, size: int = 224,
                    padding: float = .04, min_side: float = .16) -> List[Tuple[int, int, int, int]]:
    """Return pixel-space (left,top,right,bottom) boxes from patch-grid masks."""
    if mask.ndim != 3:
        raise ValueError('mask must have [B,h,w] shape')
    _, gh, gw = mask.shape
    boxes = []
    for m in mask.detach().cpu().numpy().astype(bool):
        ys, xs = np.where(m)
        if not len(xs):
            boxes.append((0, 0, size, size))
            continue
        x0, x1 = xs.min() / gw, (xs.max() + 1) / gw
        y0, y1 = ys.min() / gh, (ys.max() + 1) / gh
        cx, cy = (x0 + x1)/2, (y0+y1)/2
        wx = max(x1-x0 + 2*padding, min_side)
        wy = max(y1-y0 + 2*padding, min_side)
        x0 = max(0., cx - wx/2)
        x1 = min(1., cx + wx/2)
        y0 = max(0., cy - wy/2)
        y1 = min(1., cy + wy/2)
        # 1-pixel minimum guaranteed
        left = min(size-1, max(0, int(np.floor(x0*size))))
        top = min(size-1, max(0, int(np.floor(y0*size))))
        right = max(left+1, min(size, int(np.ceil(x1*size))))
        bottom = max(top+1, min(size, int(np.ceil(y1*size))))
        boxes.append((left, top, right, bottom))
    return boxes


def box_crops_as_clip(raw_01: torch.Tensor, boxes: List[Tuple[int,int,int,int]],
                      output_size: int = 224) -> torch.Tensor:
    """Resize each box to CLIP input. raw_01 is BEFORE CLIP normalization."""
    if raw_01.shape[0] != len(boxes):
        raise ValueError('boxes count does not match image batch')
    views = []
    for raw, (x0,y0,x1,y1) in zip(raw_01, boxes):
        crop = TF.resized_crop(raw, y0, x0, y1-y0, x1-x0,
                               [output_size, output_size],
                               interpolation=InterpolationMode.BILINEAR,
                               antialias=True)
        views.append(TF.normalize(crop, MEAN, STD))
    return torch.stack(views)


def _get_cls_attn_from_mha(mha, x: torch.Tensor) -> torch.Tensor:
    """Compute true multihead CLS attention from MHA q/k projections.

    x: the last visual MHA's input, before it applies its packed projection.
    Respects softmax over CLS + all patch keys; CLS self-mass removed then
    spatial weights renormalized. Does not alter the forward computation.
    """
    if mha.in_proj_weight is None:
        raise NotImplementedError('This extractor requires packed MHA qkv weights')
    if not mha.batch_first:  # OpenAI CLIP uses sequence-first [L,B,C]
        x = x.transpose(0, 1)
    b, tokens, dim = x.shape
    heads = mha.num_heads
    if tokens < 2 or dim % heads:
        raise ValueError('Invalid CLIP visual token/head shape')
    with torch.no_grad():
        xb = x.float()
        weights = mha.in_proj_weight.float()
        bias = mha.in_proj_bias.float() if mha.in_proj_bias is not None else None
        q = F.linear(xb[:, :1], weights[:dim], bias[:dim] if bias is not None else None)
        k = F.linear(xb, weights[dim:2*dim],
                     bias[dim:2*dim] if bias is not None else None)
        hd = dim // heads
        q = q.view(b, 1, heads, hd).permute(0,2,1,3)
        k = k.view(b, tokens, heads, hd).permute(0,2,1,3)
        score = (q * hd**-.5) @ k.transpose(-1,-2)
        weights = score.softmax(-1).mean(dim=1)[:, 0, 1:]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        grid = int(round((tokens-1)**.5))
        if grid*grid != tokens-1:
            raise ValueError(f'Non-square CLIP patch tokens: {tokens-1}')
        return weights.view(b, grid, grid)


class CLIPLastLayerAttention:
    """Temporary lightweight hook used on the ORIGINAL CLIP forward only."""
    def __init__(self, clip_model):
        visual = clip_model.visual
        blocks = visual.transformer.resblocks
        self.mha = blocks[-1].attn
        self.enabled = False
        self.maps = []
        self.handle = self.mha.register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        if self.enabled:
            self.maps.append(_get_cls_attn_from_mha(mod, args[0]).detach())

    def start(self):
        self.maps = []
        self.enabled = True

    def finish(self):
        self.enabled = False
        if not self.maps:
            raise RuntimeError('CLIP attention hook did not capture a global forward')
        return torch.cat(self.maps, dim=0)

    def close(self):
        self.enabled = False
        self.handle.remove()


@dataclass
class ObjectRegion:
    box: Tuple[int, int, int, int]
    patch_fraction: float
    box_fraction: float
    selected_heads: int
    fallback: str


def _largest_component(mask: np.ndarray):
    from scipy.ndimage import label
    labels, n = label(mask.astype(np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == int(sizes.argmax())


def _ncut_mask(cosine_affinity: torch.Tensor, saliency: torch.Tensor,
               gh: int, gw: int, topk: int = 12):
    """Sparse kNN normalized cut on Object-DINO selected-head affinity.

    With graph degeneracy, explicitly fall back to top saliency. Avoids the
    fully-connected threshold failure known for DINO high cosine similarities.
    """
    n = gh*gw
    def fallback(reason):
        a = saliency.detach().float().cpu().numpy().reshape(gh, gw)
        # Fractional top saliency followed by largest connected component.
        threshold = np.quantile(a, .65)
        fg = _largest_component(a >= threshold)
        if not (.03 <= fg.mean() <= .92):
            fg = np.zeros((gh,gw), dtype=bool)
            fg[gh//4:max(gh//4+1,3*gh//4),gw//4:max(gw//4+1,3*gw//4)] = True
            reason += '+central_fallback'
        return fg, reason
    if n < 4 or cosine_affinity.shape != (n,n):
        return fallback('invalid_graph')
    with torch.no_grad():
        affinity = torch.nan_to_num(cosine_affinity.float(), nan=-1.).clamp(-1.,1.)
        affinity = (affinity + affinity.T) / 2
        # Distances are converted into nonnegative affinities.
        weights = torch.exp((affinity - 1.) / .20)
        weights.fill_diagonal_(0.)
        k = min(topk, n-1)
        idx = weights.topk(k, dim=1).indices
        sparse = torch.zeros_like(weights)
        sparse.scatter_(1, idx, torch.gather(weights, 1, idx))
        sparse = torch.maximum(sparse, sparse.T)
        d = sparse.sum(dim=1)
        if d.min().item() <= 1e-9:
            return fallback('isolated_graph')
        d_sqrt = d.rsqrt()
        L = torch.eye(n, device=sparse.device) - (d_sqrt[:,None] * sparse * d_sqrt[None,:])
        try:
            eigval, eigvec = torch.linalg.eigh(L)
        except RuntimeError:
            return fallback('eigendecomposition_failure')
        if not torch.isfinite(eigvec).all() or eigval[1] < 1e-5:
            return fallback('degenerate_graph')
        v = eigvec[:,1].detach().cpu().numpy().reshape(gh,gw)
        a = v > np.median(v)
        b = ~a
        def fitness(c):
            frac = c.mean()
            if not .03 <= frac <= .92:
                return -100.
            border = np.r_[c[0,:],c[-1,:],c[:,0],c[:,-1]].mean()
            compact = _largest_component(c).sum() / max(1,c.sum())
            return compact - .75*border - .1*abs(frac-.35)
        fg = a if fitness(a) >= fitness(b) else b
        fg = _largest_component(fg)
        if not (.03 <= fg.mean() <= .92):
            return fallback('invalid_partition')
        return fg, ''


class DINOv2ObjectRegions:
    """Object-DINO-guided proposal model using a frozen HF DINOv2 backbone.

    Head descriptor: mean-column Q/K/V cosine-softmax map, following the
    released Object-DINO code (which uses this pooled form, unlike the paper's
    full flattened N-by-N maps). KMeans selects the cluster containing most
    final-layer heads. A kNN normalized cut on selected Q/K/V cosine affinities
    makes one box. This graph/box step is an ADAPTATION, not original output.
    """
    def __init__(self, model_id: str, *, device, clusters=12, graph_neighbors=12,
                 n_init=3, local_only=True, max_batch=4):
        try:
            from transformers import Dinov2Model
        except Exception as exc:
            raise RuntimeError('DINOv2 requires transformers: pip install transformers') from exc
        self.model_id = model_id
        self.model = Dinov2Model.from_pretrained(model_id, local_files_only=local_only)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.clusters = clusters
        self.graph_neighbors = graph_neighbors
        self.n_init = n_init
        self.max_batch = max_batch
        self.layers = []
        for layer in self.model.encoder.layer:
            attn = layer.attention
            if hasattr(attn, 'attention'):
                attn = attn.attention
            for names in [('query','key','value'), ('q_proj','k_proj','v_proj')]:
                if all(hasattr(attn, name) for name in names):
                    self.layers.append((attn, tuple(getattr(attn, n) for n in names)))
                    break
            else:
                raise RuntimeError('Unsupported transformers DINOv2 attention structure')
        if not self.layers:
            raise ValueError('DINOv2 model has no attention layers')
        self.patch_size = int(self.model.config.patch_size)
        self.heads = int(self.model.config.num_attention_heads)
        self.hidden_inputs = {}
        self.hooks = []
        for idx, (attn, _) in enumerate(self.layers):
            def record(module, inputs, idx=idx):
                self.hidden_inputs[idx] = inputs[0].detach()
            self.hooks.append(attn.register_forward_pre_hook(record))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.inference_mode()
    def _one_batch(self, raw_01: torch.Tensor) -> List[ObjectRegion]:
        b, _, height, width = raw_01.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError('DINOv2 input must be divisible by patch size')
        gh, gw = height // self.patch_size, width // self.patch_size
        self.hidden_inputs = {}
        pix = TF.normalize(raw_01.to(self.device), DINO_MEAN, DINO_STD)
        self.model(pixel_values=pix)
        if len(self.hidden_inputs) != len(self.layers):
            raise RuntimeError('DINOv2 QKV hooks did not capture all layers')
        # [B,L,H,N] head descriptors; CPU kmeans works per image.
        descriptors = []
        qkv_by_layer = []
        for idx, (_, projs) in enumerate(self.layers):
            x = self.hidden_inputs[idx]
            B, tokens, dim = x.shape
            if tokens != gh*gw+1:
                raise RuntimeError(f'DINOv2 returned {tokens} tokens, expected {gh*gw+1}. Register tokens not supported.')
            qkv = []
            for proj in projs:
                z = proj(x).float().reshape(B,tokens,self.heads,-1).permute(0,2,1,3)[:,:,1:]
                qkv.append(F.normalize(z, dim=-1))
            qkv_by_layer.append(qkv)
            desc = 0
            for z in qkv:
                sim = z @ z.transpose(-1,-2)
                # Mirrors released Object-DINO's softened similarity for descriptors.
                score = (sim * (z.shape[-1]**-.5)).softmax(-1).mean(dim=-2)
                desc = desc + score / 3.
            descriptors.append(desc)
        descriptors = torch.stack(descriptors, dim=1)  # [B,L,H,P]
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        regions = []
        for bi in range(b):
            X = descriptors[bi].reshape(-1, gh*gw).detach().cpu().numpy()
            X = StandardScaler().fit_transform(X)
            n_clusters = min(self.clusters, len(X))
            km = KMeans(n_clusters=n_clusters, n_init=self.n_init, random_state=42,
                        max_iter=150)
            labels = km.fit_predict(X).reshape(len(self.layers), self.heads)
            counts = np.bincount(labels[-1], minlength=n_clusters)
            cluster = int(counts.argmax())
            chosen = np.argwhere(labels == cluster)
            if len(chosen) == 0:
                raise RuntimeError('Object cluster selection returned no heads')
            combined = None
            saliency = None
            for layer_idx, head_idx in chosen:
                layer_idx, head_idx = int(layer_idx), int(head_idx)
                qs = qkv_by_layer[layer_idx]
                affinity = sum((z[bi,head_idx] @ z[bi,head_idx].T) / 3 for z in qs)
                # Row-softmax column mean, similar to the official head importance.
                one_sal = (affinity * (qs[0].shape[-1]**-.5)).softmax(-1).mean(0)
                combined = affinity if combined is None else combined + affinity
                saliency = one_sal if saliency is None else saliency + one_sal
            combined /= len(chosen)
            saliency /= len(chosen)
            fg, fallback = _ncut_mask(combined, saliency, gh, gw, self.graph_neighbors)
            mask = torch.from_numpy(fg).to(torch.bool).unsqueeze(0)
            box = boxes_from_mask(mask, size=height, padding=.06)[0]
            x0,y0,x1,y1 = box
            regions.append(ObjectRegion(box=box, patch_fraction=float(fg.mean()),
                                        box_fraction=float((x1-x0)*(y1-y0)/(height*width)),
                                        selected_heads=int(len(chosen)), fallback=fallback))
        self.hidden_inputs = {}
        return regions

    @torch.inference_mode()
    def boxes(self, raw_01: torch.Tensor) -> List[ObjectRegion]:
        result = []
        for part in raw_01.split(self.max_batch):
            result.extend(self._one_batch(part))
        return result
