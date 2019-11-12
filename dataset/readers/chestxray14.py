import os
import pandas as pd
import numpy as np

from skimage import io as sk_io, transform as sk_transform
from sklearn.preprocessing import MultiLabelBinarizer

class ChestXray14Reader(object):

    SPLITS = ['train', 'val']

    def __init__(self, data_dir, split, 
            train_text_file='train_val_list.txt', valid_text_file='test_list.txt', data_entry_file='Data_Entry_2017.csv', **kwargs):
        super(ChestXray14Reader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self._train_text_file = os.path.join(data_dir, train_text_file)
        self._valid_text_file = os.path.join(data_dir, valid_text_file)
        self._data_entry_file = os.path.join(data_dir, data_entry_file)
    
    def _load_metadata(self):
        md = pd.read_csv(self._data_entry_file)
        md.set_index('Image Index', inplace=True)
        md['train_or_val'] = 'T'
        self.metadata =  md

    def _load_id_list(self):
        self._id_list = {'train':[], 'val':[]}
        with open(self._train_text_file, 'r') as f:
            self._id_list['train'] = [ i.strip() for i in f.readlines()]
        with open(self._valid_text_file, 'r') as f:
            self._id_list['val'] = [ i.strip() for i in f.readlines()]

    def _get_labels(self, id):
        return self.metadata.loc[id]['Finding Labels'].split('|')

    def _encode_label_bitmap(self):
        self._labels = {}
        for sp in ['train', 'val']:
            self._labels[sp] = [self._get_labels(id) for id in self._id_list[sp]]
	
	for id in self._id_list['val']:   
	    self.metadata.loc[id, 'train_or_val'] = "V"
        
        # deal with Multi-labels 
        encoder = MultiLabelBinarizer()
        encoder.fit(self._labels['train'] + self._labels['val'])
        self._classes = encoder.classes_
       
        self._label_bitmaps = {}
        for sp in ['train', 'val']:
            self._label_bitmaps[sp] = encoder.transform(self._labels[sp])

    def labels(self):
        return self._classes.tolist()

    @property
    def total(self): 
        return len(self._id_list[self._split])

    def parse(self):
        self._load_id_list()
        self._load_metadata()
        self._encode_label_bitmap()

    def iterate(self):
        for id in self._id_list[self._split]:
            filepath = os.path.join(self._data_dir, 'images', id)
            image = sk_io.imread(filepath)
	    if image.shape != (1024, 1024):
		image = image[:,:,0]
	    image_buffer = sk_transform.resize(image, (256, 256)) * 255
	    image_buffer = image_buffer.astype(np.int8)
	    height, width = image_buffer.shape
	    image_string = image_buffer.tostring()

            labels_str = self.metadata.loc[id]['Finding Labels']

            index = self._id_list[self._split].index(id)
            labels_encoded = self._label_bitmaps[self._split][index]
            labels = self._labels[self._split][index]

            yield {
                'filepath': filepath,
                'image': image_string,
                'height': height,
                'width': width,
                'labels_encoded': labels_encoded,
                'labels_str': labels_str,
                'labels': labels,
            }
