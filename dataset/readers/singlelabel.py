import os
import json
from myelindl.utils.tfrecordreader import TFRecordReader
from myelindl.utils.image import embed_image_html

class SingleLabelReader(object):

    SPLITS = ['train', 'val']

    def __init__(self, data_dir, split, **kwargs):
        self.data_dir = data_dir
        self.split = split
        
        self.reader = TFRecordReader(self.data_dir, self.split, '%s/%s.tfrecords')

        with open(os.path.join(self.data_dir, 'classes.json')) as f:
            self.classes = json.load(f)
        self.type = 'image'

    def iterate(self, skip=0, label=None, search=None):
        if label != None:
            label = self.classes.index(label)

        generator = self.reader.parsed_entries()
        count = 0
        for img_id in range(0, self.reader.total_entries):
            item = generator.next()
            if search and str(search) not in str(img_id):
                continue

            if label is None:
                count += 1
            else:
                if str(label) == str(item['label']):
                    count += 1
                else:
                    continue

            if count <= skip:
                continue

            yield {
                'id': img_id,
                'image': item['img'],
                'label': item['text'],
            }

    def iterate_visual_item(self, skip=0, label=None, search=None):  
        for item in self.iterate(skip, label, search):
            record = {
                "tf_img_id": item['id'],
                "label": item['label'],
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
