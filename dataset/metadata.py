import os
import json

from threading import Lock
DATASET_PATH = '/data/dataset'
DATASET_METADATA_PATH = '/data/dataset/metadata.json'
DEFAULT_METADATA = {'datasets':{}}
lock = Lock()


def read_dataset_metadata():
    
    data = DEFAULT_METADATA
    with open(DATASET_METADATA_PATH, 'r') as infile:
        data = json.load(infile)
    return data

def save_dataset_metadata(metadata):
    with open(DATASET_METADATA_PATH, 'w') as outfile:
        json.dump(metadata, outfile)


def ensure_dataset_db():
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    if not os.path.exists(DATASET_METADATA_PATH):
        with open(DATASET_METADATA_PATH, 'w') as outfile:
            json.dump(DEFAULT_METADATA, outfile)



ensure_dataset_db()

