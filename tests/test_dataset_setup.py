from pathlib import Path
import tarfile
import pytest

import neglab_crop.dataset_setup as ds


def make_images(root, n, nested=True):
    folder = root / 'images' / 'category' if nested else root / 'images'
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        (folder / f'image_{k:03}.jpg').write_bytes(b'demo')
    return root / 'images'


def test_discover_and_flatten_textures(tmp_path, monkeypatch):
    monkeypatch.setitem(ds.EXPECTED, 'Textures', 4)
    src = make_images(tmp_path / 'Public' / 'DTD', 4)
    official = tmp_path / 'NegLabel'
    project = tmp_path / 'Compare'
    project.mkdir()
    found, num, origin = ds.discover('Textures', official_root=official, project_root=project,
                                      search_roots=[tmp_path / 'Public'], explicit={'Textures': src})
    assert found == src.resolve()
    assert num == 4
    dest = ds.official_image_dir(official, 'Textures')
    assert ds.prepare_flat_view(found, dest, 'Textures') == 4
    assert ds.count_images(dest, recursive=False) == 4
    assert sorted(p.name for p in dest.iterdir()) == [f'category__image_{k:03}.jpg' for k in range(4)]
    assert all(p.is_symlink() and p.resolve().is_file() for p in dest.iterdir())
    assert ds.prepare_flat_view(found, dest, 'Textures') == 4


def test_missing_and_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setitem(ds.EXPECTED, 'SUN', 3)
    src = make_images(tmp_path / 'SUN', 2, nested=False)
    found, count = ds.best_image_root(tmp_path / 'SUN', 'SUN')
    assert count == 2
    with pytest.raises(RuntimeError, match='source count'):
        ds.prepare_flat_view(found, tmp_path / 'out', 'SUN')
    assert not (tmp_path / 'out').exists()


def test_names_and_bounded_discovery(tmp_path):
    (tmp_path / 'SUN397').mkdir()
    (tmp_path / 'Places365').mkdir()
    (tmp_path / 'SUN').mkdir()
    assert ds.normalized_name('textures') == 'Textures'
    assert ds.normalized_name('INATURALIST') == 'iNaturalist'
    assert set(name for name, _ in ds.candidate_dirs(tmp_path)) == {'SUN'}


def test_archive_traversal_rejected(tmp_path):
    import io
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        data = b'bad'
        info = tarfile.TarInfo('../evil.txt')
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError, match='Unsafe archive'):
        ds._safe_extract(archive, tmp_path / 'extract')
    assert not (tmp_path / 'evil.txt').exists()


def test_prepare_then_verify_all_four(tmp_path, monkeypatch):
    import argparse
    project = tmp_path / 'Compare'
    project.mkdir()
    official = tmp_path / 'NegLabel'
    (official / 'custom_config').mkdir(parents=True)
    (official / 'custom_config' / 'clip.py').write_text('# stub\n')
    id_path = tmp_path / 'ImageNet' / 'val'
    # Keep the 1000 category structural constraint, but use 1000 small synthetic files.
    for k in range(1000):
        folder = id_path / f'n{k:08d}'
        folder.mkdir(parents=True)
        (folder / f'image{k:04d}.JPEG').write_bytes(b'fake')
    monkeypatch.setattr(ds, 'ID_EXPECTED', 1000)
    explicit = []
    for name in ds.DATASET_NAMES:
        monkeypatch.setitem(ds.EXPECTED, name, 3)
        source = make_images(tmp_path / 'sources' / name, 3)
        explicit.append(f'{name}={source}')
    args = argparse.Namespace(project_root=project, official_root=official, id_path=id_path,
                              search_root=[tmp_path / 'sources'], source=explicit,
                              action='prepare', download_missing=False, strict=True)
    assert ds.run(args) == 0
    assert (project / 'data' / 'imagenet_val_ood_index.txt').exists()
    for name in ds.DATASET_NAMES:
        output = ds.official_image_dir(official, name)
        assert ds.count_images(output, recursive=False) == 3
    args.action = 'verify'
    assert ds.run(args) == 0
