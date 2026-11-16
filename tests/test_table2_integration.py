"""Tiny end-to-end test against synthetic data, no CLIP weights or GPU."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from table2_idonly import run, report
from table2_idonly.core import DATASETS


class FakeModel:
    def eval(self):
        return self
    def requires_grad_(self, flag):
        return self
    def encode_text(self, tokens):
        x = tokens.float()
        return torch.cat([torch.sin(x/29), torch.cos(x/17), x / 250, 1/(x+1)], dim=1)
    def encode_image(self, img):
        z = img.float()
        means = z.mean(dim=(2,3))
        std = z.std(dim=(2,3))
        return torch.cat([means, std, means * std, means**2], dim=1)


def fake_load_clip(name, device):
    def tokenize(strings):
        # Each prompt maps to 3 token dimensions, pseudo-embeddings are reproducible.
        return torch.tensor([[len(s), sum(map(ord, s)) % 251, sum(map(ord,s[2:])) % 223]
                             for s in strings], dtype=torch.float32)
    return FakeModel(), tokenize


class EndToEndTests(unittest.TestCase):
    def test_runner_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            neg = root/'NegLabel'
            cn = neg/'mmcls/models/classifiers'
            cn.mkdir(parents=True)
            (cn/'class_names.py').write_text('CLASS_NAME={"imagenet": ["class_"+str(i) for i in range(1000)]}\n'
                                              'prompt_templates=["a {}"]*86\n', encoding='utf-8')
            util = neg/'mmcls/utils'
            util.mkdir(parents=True)
            (util/'ood_evaluate.py').write_text('import numpy as np\n'
                                                 'from sklearn.metrics import roc_auc_score\n'
                                                 'def evaluate_all(inside, outside):\n'
                                                 '    lab=np.r_[np.ones(len(inside)),np.zeros(len(outside))]\n'
                                                 '    s=np.r_[inside,outside]\n'
                                                 '    v=roc_auc_score(lab,s)\n'
                                                 '    return v,v,v,0.1\n', encoding='utf-8')
            roots = {}
            for d_i, ds in enumerate(DATASETS):
                folder = root/'images'/ds
                folder.mkdir(parents=True)
                roots[ds] = folder
                for i in range(3):
                    rng = np.random.default_rng(d_i*1000+i)
                    img = (rng.random((255, 260, 3))*255).astype('uint8')
                    Image.fromarray(img).save(folder / f'{i:03d}.png')
            out = root/'results'
            argv = ['--dataset','all', '--neglabel-root', str(neg), '--out', str(out),
                    '--workers', '0', '--device', 'cpu', '--batch-size', '2',
                    '--allow-count-mismatch']
            for ds in DATASETS:
                argv += ['--root-'+ds.lower(),str(roots[ds])]
            with patch.object(run, 'load_clip_model', fake_load_clip):
                self.assertEqual(run.main(argv), 0)
                # all scores validated and skipped on a re-run
                self.assertEqual(run.main(argv), 0)
            refs = root/'refs.csv'
            with refs.open('w',newline='') as fh:
                writer = csv.DictWriter(fh,fieldnames=['method','dataset','auroc','fpr95','source'])
                writer.writeheader()
                for method in ['NegLabel','Ours']:
                    for name in (*DATASETS[1:],'Average'):
                        writer.writerow({'method':method,'dataset':name,'auroc':90.4,'fpr95':20.2,'source':'test'})
            self.assertEqual(report.main(['--input',str(out),'--neglabel-root',str(neg),
                                          '--reference',str(refs),'--allow-incomplete']),0)
            with (out/'table2_long.csv').open(newline='') as fh:
                results = list(csv.DictReader(fh))
            self.assertEqual(len(results), 20)
            with (out/'ID.csv').open(newline='') as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 3)
            self.assertIn('\\begin{table*}', (out/'table2_paper.tex').read_text())
            self.assertTrue((out/'table2_deltas.csv').is_file())
            self.assertEqual(json.loads((out/'ID.meta.json').read_text())['complete'],True)


if __name__ == '__main__':
    unittest.main()
