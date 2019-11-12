import os
import gevent
from myelindl.core.dataset.publish import (
    get_dataset,
    get_dataset_path,
)
from myelindl.utils.image import embed_image_html

from .readers.yolo import YoloReader
from .readers.lidar_npy import LidarNpyReader
from .readers.multilabel import MultiLabelReader
from .readers.singlelabel import SingleLabelReader
from .readers.object_detection import BoundingBoxReader
from .readers.segmentation import SegmentationReader
import img_cache

VISUALIZE_PATH = '.vis'
VISUALIZERS = {
    'yolo': YoloReader,
    'lidar_npy': LidarNpyReader,
    'chestxray': MultiLabelReader,
    'classification': SingleLabelReader,
    'kitti': BoundingBoxReader,
    'coco': BoundingBoxReader,
    'segmentation': SegmentationReader,
}


class Visualizer(object):

    @property
    def total(self):
        pass

    @property
    def distributions(self):
        pass

    @property
    def labels(self):
        pass

    def iterate(offset):
        pass


def get_visualizer(visualizer):
    visualizer = visualizer.lower() if visualizer else ''
    if visualizer not in VISUALIZERS:
        raise ValueError('Visulaizer not found {}'.format(visualizer))
    return VISUALIZERS[visualizer]


def parse_filter(filter_input): 
    filters = {}
    try:
        for f in filter_input.split(','):
            k, v = f.split(':')
            filters[k] = v
    except Exception as e:
        return {}
    return filters


def reader_dataset_items(
    dataset_id,
    dataset_path,
    split='train',
    offset=0,
    limit=25,
    filters=None,
    search=None,
    sort=None,
    reader='file'
):
    reader_cls = get_visualizer(reader)
    filters = parse_filter(filters) 
    label = filters['label'] if 'label' in filters else None 

    reader = reader_cls(dataset_path, split, dataset_id=dataset_id)

    imgs = []
    for item in reader.iterate_visual_item(offset, label, search):
        imgs.append(item)
        gevent.sleep(0)
        if len(imgs) >= limit:
            break;

    total_entries = reader.total if not label else reader.distributions[label]
    if isinstance(reader, YoloReader) or isinstance(reader, LidarNpyReader):
        if search:
            total_entries = reader.get_total_entries(label, search)

    # type (object , yolo, classification, multiplelable, lidarnpy)
    return {
        'labels': reader.labels,
        'total_entries': total_entries,
        'images': imgs,
        'type': reader.type
    }


def file_dataset_items(
    id,
    dir='',
):
    dataset_path = get_dataset_path(id)
    target_dir = os.path.join(dataset_path, dir)
    
    files = []
    if not os.path.exists(target_dir):
        raise Exception('Directory not found. {}'.format(dir))

    if not os.path.isdir(target_dir):
        raise Exception('Not a directory. {}'.format(dir))
    
    for basename in os.listdir(target_dir):
        is_dir = os.path.isdir(os.path.join(target_dir, basename))
        filename, extension = os.path.splitext(basename)
        extension = extension.replace('.', '')
        files.append({
            'basename': basename,
            'filename': filename,
            'extension': extension,
            'type': 'dir' if is_dir else 'file',
            'path': os.path.join(dir, basename)
        })
        gevent.sleep(0)
    
    return files

def get_vis_img_path(id, img_file):
    return os.path.join(img_cache.get_vis_path(id), img_file)
    

