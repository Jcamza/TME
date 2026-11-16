"""Run: python -m unittest discover -s tests -p 'test_figure3*.py' -v"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from figure3.io import load_all
from figure3.stats import WeightedScores
from figure3.plotting import draw


def official_fpr95_ref(id_s,ood_s):
    labels=np.r_[np.ones(len(id_s),dtype=int),np.zeros(len(ood_s),dtype=int)]
    scores=np.r_[id_s,ood_s]
    inds=np.argsort(scores,kind='mergesort')[::-1]
    scores=scores[inds]; labels=labels[inds]
    distinct=np.where(np.diff(scores))[0]
    cut=np.r_[distinct,len(scores)-1]
    tp=np.cumsum(labels,dtype=np.float64)[cut]
    fp=1+cut-tp
    last=np.searchsorted(tp,tp[-1])
    sl=slice(last,None,-1)
    rec=np.r_[tp[sl]/tp[-1],1.]
    fpr=np.r_[fp[sl],0]/len(ood_s)
    j=np.argmin(np.abs(rec-.95))
    return float(fpr[j])


class Figure3StatsTest(unittest.TestCase):
    def test_official_parity_random_and_ties(self):
        rng=np.random.default_rng(48)
        for trial in range(100):
            a=rng.normal(.65,.2,96)
            b=rng.normal(.35,.3,52)
            if trial%2:
                a=np.round(a,1);b=np.round(b,1)
            w=WeightedScores(a,b)
            ia=rng.integers(0,len(a),size=len(a))
            ib=rng.integers(0,len(b),size=len(b))
            au,fp,_=w.metrics(ia,ib)
            expected_au=roc_auc_score(np.r_[np.ones(len(ia)),np.zeros(len(ib))],np.r_[a[ia],b[ib]])
            expected_fp=official_fpr95_ref(a[ia],b[ib])
            self.assertAlmostEqual(au,expected_au,places=12)
            self.assertAlmostEqual(fp,expected_fp,places=12)

    def test_wide_and_split_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            cfg={}
            for ds in ('ID','Places','Textures'):
                fp=td/f'{ds}.csv'
                with fp.open('w',newline='') as f:
                    wr=csv.DictWriter(f,fieldnames=['path','original','fivecrop'])
                    wr.writeheader()
                    for i in range(4):wr.writerow({'path':f'{ds}/{i}.jpg','original':.1*i,'fivecrop':.2*i})
                cfg[ds]={'file':str(fp),'original_col':'original','fivecrop_col':'fivecrop','key_col':'path'}
            config=td/'config.json';config.write_text(json.dumps(cfg))
            data=load_all(config_path=config,strict_counts=False)
            self.assertEqual(len(data['ID'].original),4)
            self.assertAlmostEqual(data['Places'].fivecrop[-1],.6)
            # Test paired split files with reversed crop row order.
            path=td/'Places_fivecrop.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['path','result']);w.writeheader()
                for i in reversed(range(4)):w.writerow({'path':f'Places/{i}.jpg','result':.2*i})
            cfg['Places']={'original':{'file':str(td/'Places.csv'),'column':'original','key_col':'path'},
                           'fivecrop':{'file':str(path),'column':'result','key_col':'path'}}
            config.write_text(json.dumps(cfg))
            data=load_all(config_path=config,strict_counts=False)
            self.assertTrue(np.allclose(data['Places'].fivecrop,[0,.2,.4,.6]))

    def test_plot_creation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            rng=np.random.default_rng(1)
            d={}
            for ds,n in [('ID',80),('Places',50),('Textures',45)]:
                s=rng.beta(7 if ds=='ID' else 3, 2 if ds=='ID' else 4,size=n)
                d[ds]=SimpleNamespace(original=s,fivecrop=np.clip(s+.008,1e-6,1-1e-6))
            stats={}
            for ds in ('Places','Textures'):
                stats[ds]={'original':{'auroc':90,'fpr95':30,'threshold_id95':.45},
                           'fivecrop':{'auroc':91,'fpr95':29,'threshold_id95':.46},
                           'delta_auroc_pp':1,'delta_fpr95_pp':-1}
            files=draw(d,stats,Path(td),transform='raw',bins=25)
            self.assertEqual(len(files),3)
            for f in files:self.assertTrue(Path(f).stat().st_size>1000)

if __name__=='__main__':unittest.main()
