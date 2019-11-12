import json
import csv
import os
import gevent
from glob import glob
from sets import Set
import PIL.Image
import tensorflow as tf

from myelindl.utils.image import embed_image_html

DEFAULT_IMAGE_DIR = 'images'
DEFAULT_LABEL_DIR = 'labels'
DEFAULT_PREDICT_DIR = 'predict'
DEFAULT_CLASSES_FILE = 'classes.txt'

class YoloReader(object):
    
    SPLITS = ['data']

    def __init__(self, data_dir, split, **kwargs):
        super(YoloReader, self).__init__()
        self.type = 'yolo'
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(self.data_dir, 
            kwargs.get('image_dir', DEFAULT_IMAGE_DIR))

        self.label_dir = os.path.join(self.data_dir,
            kwargs.get('label_dir', DEFAULT_LABEL_DIR))

        self.predict_dir = os.path.join(self.data_dir,
            kwargs.get('preict_dir', DEFAULT_PREDICT_DIR))

        self.classes_file = os.path.join(self.data_dir,
            kwargs.get('classes_file', DEFAULT_CLASSES_FILE))
        if not os.path.exists(self.image_dir) or \
                not os.path.exists(self.label_dir) or \
                not os.path.exists(self.classes_file):
            raise Exception('Must provide images folder, labels folder and classes.txt file')

        self.parse_predict_dir = os.path.exists(self.predict_dir)
    
        self.image_list = glob("{}/*".format(self.image_dir))

        self.classes = []
        self.class2label={}
        with open(self.classes_file, 'r') as f:
            rows = csv.reader(f)
            for i, row in enumerate(rows):
                self.classes.append(row[0])
                self.class2label[row[0]] = i

    def get_distributions(self):
        file_path = os.path.join(self.data_dir, 'distributions.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                dataset_distribution = json.loads(f.read())
        else:
            dataset_distribution = {}

        if self.split in dataset_distribution:
            return dataset_distribution[self.split]

        distribution = {}
        for item in self.iterate():
            for label in item['labels']:
                text = self.classes[label]
                if text in distribution:
                    distribution[text] += 1
                else:
                    distribution[text] = 1
            gevent.sleep(0)

        dataset_distribution[self.split] = distribution

        with open(file_path, 'w') as f:
            json.dump(dataset_distribution, f)
        return distribution

    def read_label_boxes(self, image_id):
        filename = os.path.join(self.label_dir, '{}.txt'.format(image_id))
        if not os.path.exists(filename):
            return []

        boxes = []
        with open(filename, 'r') as f:
            rows = csv.reader(f, delimiter=' ')
            for row in rows:
                data = {}
                try:
                    data['label'] = row[0]
                    data['x'] = row[1]
                    data['y'] = row[2]
                    data['w'] = row[3]
                    data['h'] = row[4]
                    boxes.append(data)
                except Exception as e:
                    raise Exception('Wrong format in label file {}.txt'.format(image_id))
        return boxes

    def read_predict_boxes(self, image_id):
        filename = os.path.join(self.predict_dir, '{}.txt'.format(image_id))
        if not os.path.exists(filename):
            return []

        boxes = []
        with open(filename, 'r') as f:
            rows = csv.reader(f, delimiter=' ')
            for row in rows:
                data = {}
                try:
                    data['label'] = row[0]
                    data['x'] = row[1]
                    data['y'] = row[2]
                    data['w'] = row[3]
                    data['h'] = row[4]
                    data['prob'] = row[5]
                    boxes.append(data)
                except Exception as e:
                    raise Exception('Wrong format in predicted result file {}.txt'.format(image_id))
        return boxes

    def get_total(self):
        return len(self.image_list)

    def get_classes(self):
        return self.classes

    def read_image(self, image_file):
        image = PIL.Image.open(image_file).convert('RGB')
        width, height = image.size
        return image, width, height, image_file

    def decode_yolo_bbox(self, bbox):
        x, y, w, h = (
                float(bbox['x']),
                float(bbox['y']),
                float(bbox['w']),
                float(bbox['h']))
        res = {
            'x': x - w/2,
            'y': y - h/2,
            'w': w,
            'h': h,
            'label': self.classes[int(bbox['label'])]
        }

        if 'prob' in bbox:
            res['prob'] = bbox['prob']
        return res

    def iterate(self, skip=0, label=None, search=None):
        skip_count = 0
        for image in self.image_list:
            image_id = os.path.basename(image).split('.')[0]
            if search and str(search) not in str(image_id):
                continue

            boxes = self.read_label_boxes(image_id)
            predicted_boxes = []
            if self.parse_predict_dir:
                predicted_boxes = self.read_predict_boxes(image_id)

            labels = list(set([int(x['label']) for x in boxes]))
            if label != None and label not in labels:
                # user Specify label filter, we shold skip if label not in all labels
                continue

            skip_count += 1
            if skip_count <= skip:
                continue

            img, width, height, filename = self.read_image(image)
            record = {
                'id' : image_id,
                'image': img,
                'boxes': boxes,
                'predicted_boxes': predicted_boxes,
                'labels': labels,
                'width' : width,
                'height': height,
                'filename': filename,
            }

            gevent.sleep(0)
            yield record

    def iterate_visual_item(self, skip=0, label=None, search=None):
        if label:
            label = self.class2label[label]
        for item in self.iterate(skip, label, search):
            record = {
                "tf_img_id": item["id"],
                "b64": embed_image_html(item['image']),
                "boxes": [self.decode_yolo_bbox(bb) for bb in item["boxes"]],
                "predicted_boxes": [self.decode_yolo_bbox(bb) for bb in item["predicted_boxes"]],
            }
            yield record

    def get_total_entries(self, label=None, search=None):
        total = 0
        if label != None:
            label = self.class2label[label]

        for f in self.image_list:
            image_id = os.path.basename(f).split('.')[0]
            if str(search) not in str(image_id):
                continue
            
            if label != None:
                boxes = self.read_label_boxes(image_id)
                labels = list(set([int(x['label']) for x in boxes]))
                if label not in labels:
                    continue
            
            total += 1

        return total

    @property
    def distributions(self):
        return self.get_distributions()

    @property
    def labels(self):
        return self.classes

    @property
    def total(self):
        return len(self.image_list)
