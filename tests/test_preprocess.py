from PIL import Image
import torch
from torchvision import transforms as T
from neglab_crop.preprocess import ImageViews, MEAN, STD


def test_global_matches_official_preprocess_exactly():
    img = Image.new('RGB', (389, 253), color=(38, 144, 216))
    official = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    views = ImageViews('fivecrop')(img)
    assert views.shape == (6, 3, 224, 224)
    assert torch.equal(views[0], official(img))


def test_tencrop():
    x = Image.new('RGB', (350, 245), color=(10, 20, 30))
    assert ImageViews('tencrop')(x).shape == (11, 3, 224, 224)
