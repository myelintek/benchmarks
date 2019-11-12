import os
import numpy as np
import time
from PIL import Image, ImageOps
from .tfrecords_writer import ProgressHook

LABEL_COLORS = [
    [0, 0, 0],
    [18, 254, 0],
    [0, 200, 254],
    [248, 0, 254],
    [63, 81, 181],
    [33, 150, 243],
    [233, 30, 99],
    [0, 150, 136],
    [159, 39, 176],
    [255, 235, 59],
    [76, 175, 80],
    [255, 193, 7],
    [139, 195, 74],
    [255, 152, 0],
    [205, 220, 57],
    [255, 87, 34],
]

CHANNELS = ['x', 'y', 'z', 'i', 'r']
MAX = np.array([80., 60., 32, 1., 80.])
MIN = np.array([0.,  0 , 0, 0., 0.])
SIZE = MAX - MIN

merge_label = lambda x, y : np.where(y > 0, y , x)
def convert_to_images(npy_array):
    height, width, channel = npy_array.shape
   
    # Prepare label image
    label = npy_array[:, :, 5]
    label = label.astype(np.uint8)
    label = label.tolist()

    for i in xrange(height):
        for j in xrange(width):
            label[i][j] = LABEL_COLORS[label[i][j] % len(LABEL_COLORS)]

    label = np.array(label)
    label = label/255.
    label = label.reshape((height, width, 1, 3))
    label = np.repeat(label, 5, axis=2)
    results = {}
    
    data = np.abs(npy_array[:, :, :5])
    normalized = (data - MIN) / SIZE 
    
    normalized = normalized.reshape((height, width, 5, 1))
    normalized = np.repeat(normalized, 3 , axis=3)
    
    labeled = merge_label(normalized, label)
    labeled = labeled * 255

    for i, ch in enumerate(CHANNELS):
        arr = labeled[:, :, i]
        results[ch] = arr.astype(np.uint8)
    
    return results

def concate_images(results):
    return Image.fromarray(np.concatenate((
        results['x'],
        results['y'],
        results['z'],
        results['i'],
        results['r'],
    ),axis=0))


class LidarImageWriter(object):
    def __init__(self, 
            reader, output_dir, split='data', 
            progress_callback=None, progress_hook=ProgressHook):

        self._output_dir = output_dir
        self._reader = reader
        self._split = split
        self._progress_hook = progress_hook
        self._progress_callback = progress_callback
        self._channels = CHANNELS

    def ensure_dir(self, d):
        d = os.path.join(
            self._output_dir,
            d
        )
        if not os.path.exists(d):
            os.makedirs(d)
            
    def save(self):
        self._reader.parse()
        for ch in self._channels:
            self.ensure_dir(ch)

        with self._progress_hook(self._reader.total, self._progress_callback, split=self._split) as bar:
            for record in self._reader.iterate():
                images = convert_to_images(record['npy_array'])
                image_name = '{}.png'.format(record['id'])

                for folder in images.keys():
                    filepath = os.path.join(
                        self._output_dir,
                        folder,
                        image_name
                    )
                    img = Image.fromarray(images[folder].astype(np.uint8))
                    img.save(filepath)

                bar.update(1)

