import os
import fnmatch
import numpy as np
import glob
import gevent
from myelindl.utils.image import embed_image_html
MAX_DISTANCE = 80

class LidarNpyReader(object):
    SPLITS = ['data']    
    def __init__(self, data_dir, split, **kwargs):
        super(LidarNpyReader, self).__init__()
        self._data_dir = data_dir
        self._split = split
        self._npy_files = []
        self.type = 'lidar_npy'
        self.parse()
    def labels(self):
        return []

    def get_distributions(self):
        file_path = os.path.join(self._data_dir, 'distributions.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                dataset_distribution = json.loads(f.read())
        else:
            dataset_distribution = {}

        if self._split in dataset_distribution:
            return dataset_distribution[self._split]

        distribution = {}
        for item in self.iterate():
            labels = item['labels']
            for label in labels:
                if label in distribution:
                    distribution[label] += 1
                else:
                    distribution[label] = 1

        dataset_distribution[self._split] = distribution

        with open(file_path, 'w') as f:
            json.dump(dataset_distribution, f)
        return distribution

    def parse(self):
        matches = []
        for root, dirnames, filenames  in os.walk(self._data_dir):
            for filename in fnmatch.filter(filenames, '*.npy'):
                matches.append(os.path.join(root, filename))

        self._npy_files = sorted(matches)
    
    def iterate(self, offset=0, label=None, search=None):
        count = 0
        for f in self._npy_files:
            filename = os.path.basename(f).split('.')[0]
            if search and str(search) not in str(filename):
                continue

            count += 1
            if count <= offset:
                continue

            raw = np.load(os.path.join(self._data_dir, f))
            labels = np.unique(raw[:, :, 5]).astype(np.uint8).tolist()

            gevent.sleep(0)
            yield {
                'id': filename,
                'filepath': f,
                'labels': labels,
                'npy_array': raw,
            }

    def convert_image(self, npy_arr):
        from myelindl.core.dataset.writers.lidar_writer import (
            convert_to_images,
            concate_images,
        )
        converted = convert_to_images(npy_arr)
        concated = concate_images(converted)
        return concated

    def iterate_visual_item(self, skip=0, label=None, search=None):

        for item in self.iterate(skip, label, search):
            record = {
                "tf_img_id": item["id"],
                "b64": embed_image_html(self.convert_image(item['npy_array'])),
            }
            yield record

    def get_total_entries(self, label=None, search=None):
        total = 0
        for f in self._npy_files:
            filename = os.path.basename(f).split('.')[0]
            if str(search) not in str(filename):
                continue
            
            if label:
                raw = np.load(os.path.join(self._data_dir, f))
                labels = np.unique(raw[:, :, 5]).astype(np.uint8).tolist()
                if str(label) not in labels:
                    continue

            total += 1

        return total

    @property
    def distributions(self):
        return self.get_distributions()

    @property
    def labels(self):
        return []

    @property
    def total(self):
        return len(self._npy_files)
