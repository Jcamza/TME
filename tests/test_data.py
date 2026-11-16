from PIL import Image
from neglab_crop.data import read_dataset_from_config, ViewsDataset
from neglab_crop.preprocess import ImageViews


def test_txt_and_folder(tmp_path):
    root = tmp_path / 'official'
    (root/'images').mkdir(parents=True)
    (root/'annotations').mkdir()
    for name in ['a.jpg', 'b.png']:
        Image.new('RGB', (300, 250), color=(120, 80, 20)).save(root/'images'/name)
    (root/'annotations'/'list.txt').write_text('a.jpg 12\nb.png 9\n')
    samples = read_dataset_from_config(root, {
        'type':'TxtDataset', 'path':'images', 'data_ann':'annotations/list.txt'})
    assert len(samples) == 2
    assert [s.label for s in samples] == [12,9]
    ds = ViewsDataset(samples, ImageViews('fivecrop'))
    views, label, path = ds[0]
    assert views.shape == (6, 3, 224, 224) and label == 12
    folders = read_dataset_from_config(root, {'type':'FolderDataset', 'path':'images'})
    assert len(folders) == 2
