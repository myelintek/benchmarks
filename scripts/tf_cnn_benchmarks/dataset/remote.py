import os
import json
import uuid
import requests
from download import DSDownloader
from myelindl.env import RUNS_DIR, RUNS_PREFIX
from metadata import DATASET_PATH
from myelindl.log import logger
from myelindl.webapp import scheduler


REMOTE_DS_INDEX = 'remote_datasets.json'
REMOTE_DS_PATH = '/data/dataset'
REMOTE_INDEX_URL = (
    'http://140.96.29.39:8000/dataset/remote_datasets.json'
)


def read_remote_db():
    """Reads the checkpoints database file from disk."""
    path = os.path.join(REMOTE_DS_PATH, REMOTE_DS_INDEX)
    if not os.path.exists(REMOTE_DS_PATH):
        os.mkdir(REMOTE_DS_PATH)
    if not os.path.exists(path):
        fetch_remote_index()
        if not os.path.exists(path):
            return {}

    with open(path) as f:
        index = json.load(f)

    return index


def download(name):
    db = read_remote_db()
    url = ''
    for n, d in db.iteritems():
        if n == name:
            url = d['url']
    download_remote_ds(db, name, url)


def download_remote_ds(db, name, url):
    path = os.path.join(DATASET_PATH, name)
    DSDownloader(str(url), str(path))


def fetch_remote_index():
    url = REMOTE_INDEX_URL
    logger.info('Retrieving remote index... ')
    db = requests.get(url).json()
    logger.info('done.')
    save_ds_db(db)
    return db


def save_ds_db(ds):
    """Overwrites the database file in disk with `datasets`."""
    path = os.path.join(REMOTE_DS_PATH, REMOTE_DS_INDEX)
    with open(path, 'w') as f:
        json.dump(ds, f)
