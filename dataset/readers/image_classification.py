import os
import random
import tensorflow as tf
import six
from collections import OrderedDict
from PIL import Image


SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.pgm')
CLASS_FILE = 'classes.json'

def read_image(path):
    with tf.gfile.GFile(path, 'rb') as f:
        image = f.read()
    return image

SPLIT_FOLDER = {
    'train': 'train',
    'val'  : 'test',
}

class ClassiReader(object):
    SPLITS = ['train', 'val']
    def __init__(self, data_dir, split, **kwargs):
        super(ClassiReader, self).__init__(**kwargs)
        self._split = split
        self._data_dir = data_dir
        self._distribution = OrderedDict()
        self._file_paths = []
        self._file_texts = []
        self._labelid = {} # {'dog':0,'cat':1,'pig':2}

    def parse(self):
        subdirs = []
        split_dir = os.path.join(self._data_dir, SPLIT_FOLDER[self._split])
        if os.path.exists(split_dir) and os.path.isdir(split_dir):
            for filename in os.listdir(split_dir):
                subdir = os.path.join(split_dir, filename)
                if os.path.isdir(subdir):
                    subdirs.append(subdir)
        else:
            raise ValueError('folder does not exist')
        subdirs.sort()

        if len(subdirs) < 2:
            raise ValueError('folder must contain at least two subdirectories')

        label_index = 0
        for subdir in subdirs:
            # Use the directory name as the label
            label_name = subdir
            label_name = os.path.basename(label_name)
            if label_name.endswith('/'):
                # Remove trailing slash
                label_name = label_name[0:-1]
            if label_name not in self._distribution:
                self._distribution[label_name] = 0
            if label_name not in self._labelid:
                self._labelid[label_name] = label_index

            # Read all images in the subdir/label_name folder
            for dirpath, _, filenames in os.walk(os.path.join(split_dir, subdir), followlinks=True):
                for filename in filenames:
                    if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                        self._file_paths.append('%s' % (os.path.join(split_dir, subdir, dirpath, filename)))
                        self._file_texts.append(label_name)
                        self._distribution[label_name] += 1
            label_index += 1

    def labels(self):
        return list(self._distribution.keys())

    def distribution(self):
        return self._distribution
    
    @property
    def total(self):
        return len(self._file_paths)

    def get_shuffle_paths(self):
        shuffled_index = list(range(len(self._file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)
        filepaths = [self._file_paths[i] for i in shuffled_index]
        texts = [self._file_texts[i] for i in shuffled_index]
        labels = [self._labelid[text] for text in texts]
        return filepaths, labels, texts

    def iterate(self):
        filepaths, labels, texts = self.get_shuffle_paths()
        for i in range(self.total):
            filepath = filepaths[i]
            label = labels[i]
            text = texts[i]
            image = read_image(filepath)

            image_pil = Image.open(six.BytesIO(image))
            width = image_pil.width
            height = image_pil.height
            
            yield {
                'filepath': filepath,
                'label': label,
                'text' : text,
                'image_buffer': image,
                'width': width,
                'height': height,
            }
 
