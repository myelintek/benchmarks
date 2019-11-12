import os
from myelindl.core.dataset.publish import (
    get_dataset_path,
)
VIS_PATH = '.vis'


def file_create_date(path):
    stat = os.stat(path)
    try:
        return stat.st_birthtime
    except AttributeError:
        return stat.st_mtime


def format_img_url(dataset_id, img_file):
    return '/api/datasets/{}/items/img/{}'.format(dataset_id, img_file)


def get_vis_path(dataset_id):
    return os.path.join(
        get_dataset_path(dataset_id),
        VIS_PATH,
    )

def img_cache_get(dataset_id, img_file, time_after=None):
    vis_path = get_vis_path(dataset_id)
    img_path = os.path.join(vis_path, img_file)
    if not os.path.exists(img_path):
        return None

    if time_after and file_create_date(img_path) < time_after:
        return None

    return format_img_url(dataset_id, img_file)
  
def img_cache_put(dataset_id, img_file, img, optimize=True, quality=90):
    vis_path = get_vis_path(dataset_id)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    img_path = os.path.join(vis_path, img_file)
    img.save(img_path, optimize=optimize, quality=quality)

    return format_img_url(dataset_id, img_file)

