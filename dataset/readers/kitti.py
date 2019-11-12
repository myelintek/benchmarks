import json
import csv
import os
from glob import glob
from sets import Set
import PIL.Image
import tensorflow as tf

from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.dataset import read_image

# create val dataset from train dataset
NUM_VALIDATION=500


class KittiReader(ObjectDetectionReader):
    
    SPLITS = ['train', 'val']

    def __init__(self, data_dir, split,
                 use_supercategory=False, **kwargs):
        super(KittiReader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self.training_image_dir = os.path.join(self._data_dir, 'training', 'image_2') 
        self.training_label_dir = os.path.join(self._data_dir, 'training', 'label_2') 

        images = glob("%s/*" % self.training_image_dir)
        labels = glob("%s/*.txt" % self.training_label_dir)

        tf.logging.info('{} images files {}'.format(self.training_image_dir, len(images)) )
        tf.logging.info('{} label files {}'.format(self.training_label_dir, len(labels)) )
        image_ids = [ os.path.basename(x).split('.')[0] for x in images ]
        label_ids = [ os.path.basename(x).split('.')[0] for x in labels ]

        imageSet = Set(image_ids)
        labelSet = Set(label_ids)

        difference = imageSet.symmetric_difference(labelSet)

        if difference :
            logger.warning('Kitti format data set. some image and label are not match. Check these id %s' % difference)

        valid_image_list = list( imageSet.intersection(labelSet))

        size = len(valid_image_list)
        validation_start_point = size - int(NUM_VALIDATION)
        self.subsets = {
                'train': valid_image_list[:validation_start_point],
                'val': valid_image_list[validation_start_point:]}

        if self._split == 'train':
            self._total_records = len(self.subsets['train'])
        elif self._split == 'val':
            self._total_records = len(self.subsets['val'])
        tf.logging.info('Total {} images: {}'.format(self._split, self._total_records))

        category_list = []
        for image_id in self.subsets[self._split]:
            objs = self._read_kitti_labels(image_id)
            for obj in objs:
                category_list.append(obj['type'])
        self._total_classes = sorted(set(category_list))

        self.yielded_records = 0
        self.errors = 0
        self.label_id_map = {}

    def _read_kitti_labels(self, image_id):
        filename = os.path.join(self.training_label_dir, '%s.txt' % image_id  )
        objects = []
        with open(filename, 'r') as lf:
            rows = csv.reader(lf, delimiter=' ')
            for row in rows:
                objects.append({
                    'type' : row[0],
                    'truncated': float(row[1]),
                    'occluded': int(row[2]),
                    'alpha': float(row[3]),
                    'bbox': [float(x) for x in row[4:8]],
                    'dimensions': [float(x) for x in row[8:11]],
                    'location': [float(x) for x in row[11:14]],
                    'rotation_y': float(row[14])
                    })

        return objects

    def get_total(self):
        return self._total_records

    def get_classes(self):
        return self._total_classes

    def labels(self):
        return self._total_classes

    def _get_label_id(self, label):
        label_id = self._total_classes.index(label)
        return label_id

    def _read_kitti_image(self,  image_id):
        filename = os.path.join(self.training_image_dir, '%s.png' % image_id)
        image = PIL.Image.open(filename).convert('RGB')
        width, height = image.size
        with tf.gfile.GFile(filename, 'rb') as f:
            image = f.read()
        return image, width, height, filename

    def iterate(self):
        for image_id in self.subsets[self._split]:
            if self._stop_iteration():
                return

            objs = self._read_kitti_labels(image_id)

            gt_boxes = []
            label_set = set()
            for obj in objs:
                obj_b = obj['bbox']
                gt_boxes.append({
                    'xmin': obj_b[0],
                    'ymin': obj_b[1],
                    'xmax': obj_b[2],
                    'ymax': obj_b[3],
                    'label': self._get_label_id(obj['type']),
                })
                label_set.add(obj['type'])
                
            if len(gt_boxes) == 0:
               continue

            image, width, height, image_file = self._read_kitti_image(image_id)
            record = {
                'width': width,
                'height': height,
                'depth': 3,
                'filename': os.path.basename(image_file),
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }
            self._will_add_record(record)
            self.yielded_records += 1

            yield record
