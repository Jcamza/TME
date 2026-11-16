"""End-to-end paired-data/score smoke using a tiny FAKE encoder (not benchmark scores)."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from PIL import Image
from neglab_crop.data import Sample
from neglab_crop.preprocess import ImageViews
from neglab_crop.runner import _collect_one


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1))
    def encode_image(self, x):
        # deterministic Bx6 fake embeddings for inexpensive pipeline validation
        a = x.mean(dim=(2,3))
        return torch.cat([a, a[:, :2], a.sum(dim=-1, keepdim=True)], dim=-1)


class FakeOfficial:
    def __init__(self):
        self.classifier = SimpleNamespace(model=TinyEncoder(),
            text_features_pos=F.normalize(torch.randn(5, 6), dim=-1),
            text_features_neg=F.normalize(torch.randn(20, 6), dim=-1))
        self.ngroup = 5
        self.group_fuse_num = None
        self.t = 1
    def grouping(self, pos, neg, num, ngroup, random_permute):
        from neglab_crop.scoring import neglabel_score_reference
        return neglabel_score_reference(pos, neg, t=self.t, ngroup=ngroup)
    def forward(self, img):
        from neglab_crop.runner import _normalize_like_official, _official_score
        h = _normalize_like_official(self.classifier.model.encode_image(img))
        return _official_score(self, h)


def test_paired_runner_creates_scores_and_alpha0_matches_original(tmp_path):
    samples = []
    for i in range(3):
        img_path = tmp_path / f'{i}.png'
        Image.new('RGB', (320, 260), (i*60+20, i*40+40, 100)).save(img_path)
        samples.append(Sample(img_path, i))
    args = SimpleNamespace(batch_size=2, workers=0, view_batch_size=4, fusion='feature',
                           check_official_parity=True)
    scores, _ = _collect_one('TEST', samples, ImageViews(), FakeOfficial(),
                             [0.,0.2], args, tmp_path)
    assert len(scores['official_original']) == 3
    assert abs(scores['official_original'] - scores['feature_alpha_0.000']).max() < 1e-6
    assert (tmp_path / 'scores/TEST.csv').is_file()
