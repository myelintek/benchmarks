import json
import csv
import os
import gevent
from glob import glob
from sets import Set
import PIL.Image
import tensorflow as tf
from collections import OrderedDict
from myelindl.utils.image import embed_image_html
from myelindl.core.dataset import img_cache

DEFAULT_IMAGE_DIR = 'images'
DEFAULT_LABEL_DIR = 'segments'
DEFAULT_PREDICT_DIR = 'predict'
DEFAULT_CLASSES_FILE = 'classes.txt'


class SegmentationReader(object):
    
    SPLITS = ['data']

    def __init__(self, data_dir, split, **kwargs):
        super(SegmentationReader, self).__init__()
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
                not os.path.exists(self.label_dir):
            raise Exception('Must provide images folder, segments folder and predict file')
        self.dataset_id = kwargs.get('dataset_id', None)

        self.parse_predict_dir = os.path.exists(self.predict_dir)
    
        self.image_list = glob("{}/*".format(self.image_dir))
        
        self.image_list.sort()

        self.records = OrderedDict()
        for img_file in self.image_list:
            img_id = os.path.basename(img_file).split('.')[0]
            gevent.sleep(0)
            self.records[img_id] = {
                'image': img_file
            }
        # parse label
        self.label_count = 0
        for label_file in glob("{}/*.png".format(self.label_dir)):
            img_id = os.path.basename(label_file).split('.')[0]
            gevent.sleep(0)
            if img_id in self.records:
                self.records[img_id]['label'] = label_file
                self.label_count += 1
        # parse predict
        self.predict_count = 0
        self.both_count = 0
        for predict_file in glob("{}/*.png".format(self.predict_dir)):
            img_id = os.path.basename(predict_file).split('.')[0]
            gevent.sleep(0)
            if img_id in self.records:
                self.records[img_id]['predict'] = predict_file
                self.predict_count += 1
                if 'label' in self.records[img_id]:
                    self.both_count += 0

        self.type = 'segmentation'

    def render_segments(self, img, seg): 
        mask = seg.copy().convert('L')
        img.paste(seg, mask=mask)
        return img
    
    def read_image(self, image_file):
        image = PIL.Image.open(image_file)
        width, height = image.size
        return image, width, height, image_file

    def iterate(self, skip=0, label=None, search=None):
        cnt = 0
        for image_id, record in self.records.iteritems():
            cnt += 1
            if cnt <= skip:
                continue
            image_file = record['image']
            img, width, height, filename = self.read_image(image_file)
            gevent.sleep(0)
            record = {
                'id' : image_id,
                'image': img,
                'width' : width,
                'height': height,
                'filename': filename,
                'label_file': record['label'] if 'label' in record else None,
                'predict_file': record['predict'] if 'predict' in record else None,
            }

            yield record
    
    def iterate_visual_item(self, skip=0, label=None, search=None):
        show_gt = True
        show_pred = False
        if label and label in self.labels:
            show_gt = 'ground truth' == label or 'both' == label
            show_pred = 'prediction' == label or 'both' == label
 
        for item in self.iterate(skip, label, search):
            img = item['image']
            id = item['id']

            cache_img = '{}_{}_{}.jpg'.format(id, show_gt, show_pred)
            
            modify_time = 0

            if show_gt and item['label_file']:
                modify_time = img_cache.file_create_date(item['label_file'])

            if show_pred and item['predict_file']:
                modify_time = max(img_cache.file_create_date(item['predict_file']), modify_time)

            img_url = img_cache.img_cache_get(
                self.dataset_id, 
                cache_img,
                time_after=modify_time
            )
            if not img_url:
                if show_gt and item['label_file']:
                    label_file = item['label_file']
                    gt_img, _, _, _ = self.read_image(label_file)
                    img = self.render_segments(img, gt_img)

                if show_pred and item['predict_file']:
                    predict_file = item['predict_file']
                    pred_img, _, _, _ = self.read_image(predict_file)
                    img = self.render_segments(img, pred_img)
                img_url = img_cache.img_cache_put(self.dataset_id, cache_img, img)

            gevent.sleep(0)
            record = {
                "tf_img_id": item["id"],
                "b64": img_url,
            }
            yield record

    @property
    def distributions(self):
        return {
            'ground truth': self.total,
            'prediction': self.total,
            'both': self.total,
        }

    @property
    def labels(self):
        return ['ground truth', 'prediction','both']

    @property
    def total(self):
        return len(self.image_list)
