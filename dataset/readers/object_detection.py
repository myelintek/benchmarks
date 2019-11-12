import os
import json
from myelindl.utils.tfrecordreader import LumiObjTFRecordReader
from myelindl.utils.image import embed_image_html

class BoundingBoxReader(object):

    SPLITS = ['train', 'val']

    def __init__(self, data_dir, split, **kwargs):
        self.data_dir = data_dir
        self.split = split
        
        self.reader = LumiObjTFRecordReader(self.data_dir, self.split)

        with open(os.path.join(self.data_dir, 'classes.json')) as f:
            self.classes = json.load(f)

        self.type = 'object'

    def iterate(self, skip=0, label=None, search=None):
        if label != None:
            label = self.classes.index(label)
        cnt = 0
        for item in self.reader.parsed_entries():
            cnt += 1
            if cnt < skip:
                continue

            if label != None and label not in item['labels']:
                continue

            yield {
                'id': cnt,
                'image': item['img'],
                'boxes': item['boxes'],
            }


    def iterate_visual_item(self, skip=0, label=None, search=None):  
        for item in self.iterate(skip, label, search):
            for b in item["boxes"]:
                b["label"] = self.classes[int(b["label"])]
 
            record = {
                "tf_img_id": item['id'],
                "boxes": item['boxes'],
                "b64": embed_image_html(item['image'])
            }
            
            yield record
    
    @property
    def distributions(self):
        dist = {}
        for k, v in self.reader.get_distributions().iteritems():
            dist[self.classes[int(k)]] = v
        return dist

    @property
    def labels(self):
        return self.classes

    @property
    def total(self):
        return self.reader.total_entries
