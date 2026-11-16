"""Tests do not require installing OpenAI CLIP or downloading image/model weights."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from table2_idonly.core import PairedViews, feature_fusion, l2_normalize, mcm_from_features
from table2_idonly.run import inspect_existing_csv, sha256_lines


class Table2MathTests(unittest.TestCase):
    def test_global_alpha0_and_weighted_crop_mean(self):
        h0 = torch.tensor([[1., 0., 0.]])
        crops = torch.tensor([[[0., 1., 0.], [0., 1., 0.], [1., 0., 0.],
                               [1., 0., 0.], [1., 0., 0.]]])
        expected = l2_normalize(0.3*h0 + 0.7*l2_normalize(crops).mean(1))
        self.assertTrue(torch.allclose(feature_fusion(h0, crops, 0.7), expected, atol=1e-7))
        self.assertTrue(torch.allclose(feature_fusion(h0, crops, 0), l2_normalize(h0)))
        wrong = l2_normalize(0.3*h0 + 0.7*l2_normalize(l2_normalize(crops).mean(1)))
        self.assertFalse(torch.allclose(expected, wrong, atol=1e-4))

    def test_mcm(self):
        positive_text = torch.eye(3)
        z = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        expected = torch.softmax(100*z @ positive_text.T, dim=1).amax(dim=1)
        actual = mcm_from_features(z, positive_text)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.all(actual >= 1/3))
        self.assertTrue(torch.all(actual <= 1))

    def test_transforms_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.zeros((320, 480, 3), dtype=np.uint8)
            a[:, :, 0] = np.arange(480, dtype=np.uint16)[None, :] * 255 // 479
            a[:, :, 1] = np.arange(320, dtype=np.uint16)[:, None] * 255 // 319
            (root/'a').mkdir()
            img = root/'a'/'1.jpg'
            Image.fromarray(a).save(img)
            ds = PairedViews(root, [img])
            name, g, c = ds[0]
            self.assertEqual(name, 'a/1.jpg')
            self.assertEqual(g.shape, (3,224,224))
            self.assertEqual(c.shape, (5,3,224,224))
            self.assertFalse(torch.allclose(g, c[-1], atol=1e-5))
            csv_path = root/'scores.csv'
            with csv_path.open('w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=['path', 'mcm_global', 'mcm_fivecrop'])
                writer.writeheader()
                writer.writerow({'path': name, 'mcm_global':'0.8', 'mcm_fivecrop':'0.9'})
            self.assertEqual(inspect_existing_csv(csv_path, [img], root), 1)


if __name__ == '__main__':
    unittest.main()
