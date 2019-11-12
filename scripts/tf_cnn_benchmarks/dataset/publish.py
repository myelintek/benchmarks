from __future__ import print_function
import os
import uuid
import shutil
import getpass
import metadata
import traceback
import json
import sys
import time

import time
from datetime import datetime
from myelindl.core.util import get_folder_size, sizeof_fmt
from myelindl.client import MyelindlApi, MyelindlApiError
from myelindl.dataset.job import DatasetJob
#class Dataset(object):
#    def _init_(self):
#        self.id = id
#        self.name = name
#        self.description =
#        self.create_time
#        self.update_time
#        self.type
#        self.owner
    

class DatasetJobAdaptor(DatasetJob):
    """DatasetJobAdaptor
    
      Provide adaptor for myelindl dataset to digits dataset job
    
    """
    def __init__(self, dataset_dict):
        #super(DatasetJobAdaptor, self).__init__(
        #    dataset_dict['description'],
        #    name=dataset_dict['name'], 
        #    username=dataset_dict['username'])
        
        self._id = dataset_dict['id']
        self._dataset = dataset_dict 
        self.group = dataset_dict['name']
        self._name = dataset_dict['name']
        self.username = dataset_dict['username']
        self.persistent = True
        self.tasks = []
        self._dir = os.path.join("/data/dataset", self._id)
        _str_ctime = dataset_dict['create_time']
        _ctime = time.mktime(datetime.strptime(_str_ctime, "%Y-%m-%d %H:%M:%S").timetuple())
        self.status_history = [('D', _ctime)]
        self.exception = None
        self._notes = None

    def id(self):
        return self._id

    def name(self):
        return self._dataset['name']

    def json_dict(self, verbose=False):
        tags = self._dataset['tags'] if 'tags' in self._dataset else {}
        return {
            'id': self._dataset['id'],
            'name': self._dataset['name'],
            'notes': self._dataset['description'],
            'status': 'Done',
            'submitted': self._dataset['create_time'],
            'type': self._dataset['from'],
            'source': self._dataset['source'],
            'record_type': self._dataset['type'],
            'user': self._dataset['username'],
            'size': sizeof_fmt(self._dataset['size']),
            'tags': tags,
        }


def get_dataset_labels(id):
    """get_dataset_labels

    :param id: published dataset id
    :returns: the labels of this dataset
    """
    labels=[]
    with open(os.path.join(get_dataset_path(id), 'classes.json')) as f:
        labels = json.load(f)

    return labels


def get_dataset_path(id):
    """get_dataset_path

    :param id: published dataset id
    :returns: the aboslutly data set path of this dataset
    """
    return os.path.join(metadata.DATASET_PATH, id)


def list_datasets(username=None):
    """list_datasets

    :param username: list dataset by this user, if username is None will list all dataset
    :returns: list of dataset dict
    """
    data = metadata.read_dataset_metadata()
    if username:
        return data['datasets'].values()
    else:
        return [d for d in data['datasets'].values() if d['username'] == username]


def get_dataset(id):
    """get_dataset

    :param id: Published dataset id.
    :returns: dataset dict of this id.
    """
    data = metadata.read_dataset_metadata() 
    if id not in data['datasets']:
        return None
    return data['datasets'][id]


def publish_dataset(username, id, dataset):
    """publish_dataset

    :param username: Which user publish this dataset.
    :param id: ID of this dataset. shoud be uniq in /data/dataset folder
    :param dataset: dataset set dict include {'id':, 'name', 'description':, 'type':}
    :raises: Exception when required file not in dataset folder or permission deny
    """

    if username == 'root':
        temp_dir = os.path.join('/root', dataset['data_dir'])
    else:
        temp_dir = os.path.join('/home', username, dataset['data_dir'])
    
    if not os.path.exists(temp_dir):
        raise Exception('folder not found {}'.format(temp_dir))

    if id in [None, '']:
        while True:
            id = str(uuid.uuid1())[:8]
            dataset_path = get_dataset_path(id)
            if not os.path.exists(dataset_path):
                break

    dataset_path = get_dataset_path(id)
    # check folder and metadata structure
    # Copy the dataset to public dataset folder
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    shutil.move(temp_dir, dataset_path)
    if 'username' not in dataset or dataset['username'] in ['', None]:
        raise Exception('Username did not provided.')

    if dataset['type'] == 'object':
        required_files = ['train.tfrecords', 'val.tfrecords', 'classes.json']
        for f in required_files:
            if not os.path.exists(os.path.join(dataset_path, f)):
                raise Exception('Requied file not exists. {}'.format(f))
    elif dataset['type'] == 'file':
        if not os.listdir(dataset_path):
            raise Exception('Dataset are empty.')

    dataset['size'] = get_folder_size(dataset_path)

    current = str(datetime.now())
    dataset['id'] = id
    dataset['update_time'] = current
    dataset['is_published'] = True
    dataset['dataset_dir'] = dataset_path
    dataset['source'] = 'local'
    dataset['from'] = 'CLI'

    # update metadata
    with metadata.lock:
        data = metadata.read_dataset_metadata() 
        if id in data['datasets'] and data['datasets'][id]['username'] != username:
            raise Exception('Permission deny, this dataset is not belongs to you')
        
        if id in data['datasets']:
            data['datasets'][id].update(dataset)
        else:
            dataset['create_time'] = current
            data['datasets'][id] = dataset
            
        metadata.save_dataset_metadata(data)
    return id

def unpublish_dataset(username, id):
    """unpublish_dataset

    :param username: username who want to unpublish this datset
    :param id: target dataset id
    :raises: Excetion when permission deny or dataset not found.
    """

    with metadata.lock:
        data = metadata.read_dataset_metadata()
        if id not in data['datasets']:
            raise Exception('Dataset not found {}'.format(id))
        if data['datasets'][id]['username'] != username:
            raise Exception('Permission deny, this dataset is not belongs to you.')

        dataset_path = get_dataset_path(id)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        del data['datasets'][id]
        metadata.save_dataset_metadata(data)
