import os
import re
import metadata
from subprocess import check_output
from datetime import datetime
from myelindl.core.const import DATA_ROOT


def decode_id(id):
    """ data set id input might start with bk/ for bucket access
    """
    if id.startswith('bk/'):
        return id.replace('bk/', '')
    return id


def get_dataset_path(id):
    """get_dataset_path

    :param id: published dataset id
    :returns: the aboslutly data set path of this dataset
    """
    id = decode_id(id)
    return os.path.join(metadata.DATASET_PATH, id)


"""
def load():
    output = check_output("mc ls mlsteam", shell=True)
    for line in output.splitlines():
            match = re.match('\[(.*)\]\s+(\d+B) (\w+)', line)
            if match:
                bk_time = match.group(1)
                bk_name = match.group(3)
                add('system', id=bk_name, create_time=bk_time)
"""


def list():
    dataset_list = []
    # update metadata
    with metadata.lock:
        data = metadata.read_dataset_metadata()
        return data['datasets']



def check_exist(id):
    id = decode_id(id)
    with metadata.lock:
        data = metadata.read_dataset_metadata()
        if id not in data['datasets']:
            raise ValueError('dataset not found')


def update_metadata(id, metadata):
    with metadata.lock:
        data = metadata.read_dataset_metadata()

        if id not in data['datasets']:
            raise ValueError('Datatset Not found')
        if 'id' in metadata:
            raise ValueError('id can not be update')

        data['datasets'][id].update(metadata)
        metadata.save_dataset_metadata(data)


def add_tag(id, tag):
    with metadata.lock:
        data = metadata.read_dataset_metadata()

        if id not in data['datasets']:
            raise ValueError('Datatset Not found')

        if 'tags' not in data['datasets'][id]:
            data['datasets'][id]['tags'] = {}

        data['datasets'][id]['tags'].update(tag)
        metadata.save_dataset_metadata(data)


def remove_tag(id, tag):
     with metadata.lock:
        data = metadata.read_dataset_metadata()

        if id not in data['datasets']:
            raise ValueError('Datatset Not found')

        if 'tags' not in data['datasets'][id]:
            return

        del data['datasets'][id]['tags'][tag]
        metadata.save_dataset_metadata(data)
   

def add(username, id, vol_path='', create_time=None):
    # update metadata
    with metadata.lock:
        data = metadata.read_dataset_metadata()
        if not id:
            print('bucket without name.')
            raise ValueError('bucket without name.')
        if len(id) < 3:
            print('Bucket name cannot be smaller than 3 characters')
            raise ValueError('Bucket name cannot be smaller than 3 characters')
        if len(id) > 63:
            print('Bucket name cannot be greater than 63 characters')
            raise ValueError('Bucket name cannot be greater than 63 characters')
        match = re.match(r'[a-zA-Z0-9]+', id)
        if match:
            if match.group() != id:
                print('Bucket name contains invalid characters')
                raise ValueError('Bucket name contains invalid characters')
        else:
            print('Bucket name contains invalid characters')
            raise ValueError('Bucket name contains invalid characters')
        if id in data['datasets']:
            print('bucket {} already exist'.format(id))
            raise Exception('Dataset {} already exists'.format(id))
        if create_time:
            c_time = create_time
        else:
            c_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dst_path = os.path.join(DATA_ROOT, "dataset", id)
        if vol_path and not os.path.exists(vol_path):
            raise Exception('local path {} not exists'.format(vol_path))
        if os.path.exists(vol_path):
            print("create link {} -> {}".format(vol_path, dst_path))
            os.symlink(vol_path, dst_path)
        if not os.path.exists(dst_path):
            check_output("mc mb mlsteam/{}".format(id), shell=True)
        dataset = {
            'id': id,
            'name': id,
            'description': '',
            'username': username,
            'from': 'CLI',
            'source': 'local',
            'type': 'file',
            'size': 0,
            'data_dir': get_dataset_path(id),
            'create_time': c_time,
            'vol_path': dst_path if vol_path else '',
        }
        data['datasets'][id] = dataset
        metadata.save_dataset_metadata(data)


def delete(username, id):
    with metadata.lock:
        data = metadata.read_dataset_metadata()
        if id not in data['datasets']:
            raise Exception('Dataset not found {}'.format(id))
        bk_path = os.path.join(DATA_ROOT, "dataset", id)
        if 'vol_path' in data['datasets'][id]:
            vol_path = data['datasets'][id]['vol_path']
            if os.path.exists(vol_path) and os.path.islink(vol_path):
                os.unlink(vol_path)
        if os.path.exists(bk_path):
            check_output("mc rb --force mlsteam/{}".format(id), shell=True)
        del data['datasets'][id]
        metadata.save_dataset_metadata(data)
