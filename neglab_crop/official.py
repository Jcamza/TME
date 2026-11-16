"""Load **unmodified** NegLabel modules from an existing official checkout.

Avoids copying official code and avoids changing its labels or grouping algorithm.
"""
import os
import sys
from pathlib import Path


def load_official(root: Path, config: str, *, arch_override=None):
    root = Path(root).expanduser().resolve()
    expected = [root / 'mmcls/models/classifiers/multi_modal.py',
                root / 'mmcls/models/ood/clip_ood.py', root / 'txtfiles']
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Official NegLabel checkout is missing. Expected:\n' +
            '\n'.join(str(p) for p in missing) +
            '\nDownload https://github.com/tmlr-group/NegLabel and set OFFICIAL_ROOT.'
        )
    os.environ.setdefault('LOCAL_RANK', '0')
    sys.path.insert(0, str(root))
    import mmcv
    from mmcls.models import build_ood_model
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    cfg = mmcv.Config.fromfile(str(config_path))
    # Original repo contains relative wordnet paths; execution cwd must not matter.
    mcfg = dict(cfg.model)
    mcfg['classifier'] = dict(mcfg['classifier'])
    clscfg = mcfg['classifier']
    wordnet = Path(clscfg['wordnet_database'])
    if not wordnet.is_absolute():
        clscfg['wordnet_database'] = str((root / wordnet).resolve())
    if arch_override:
        # clip.load accepts its own registered backbone name or a local TorchScript/.pt file.
        clscfg['arch'] = arch_override
    model = build_ood_model(mcfg)
    model.eval()
    return cfg, model
