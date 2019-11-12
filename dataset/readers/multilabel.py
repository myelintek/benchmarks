import os
import json
from myelindl.utils.tfrecordreader import MultiLabelTFRecordReader
from myelindl.utils.image import embed_image_html

class MultiLabelReader(object):

    SPLITS = ['train', 'val']

    def __init__(self, data_dir, split, **kwargs):
        self.data_dir = data_dir
        self.split = split
        
        self.reader = MultiLabelTFRecordReader(self.data_dir, self.split, '%s/%s.tfrecords')

        with open(os.path.join(self.data_dir, 'classes.json')) as f:
            self.classes = json.load(f)
        self.type = 'multilabel'

    def iterate(self, skip=0, label=None, search=None):
        generator = self.reader.parsed_entries()
        count = 0
        for img_id in range(0, self.reader.total_entries):
            item = generator.next()
            if search and str(seatch) not in str(img_id):
                continue

            labels = item['text'].split('|')
            if label is None:
                count += 1
            else:
                if str(label) in labels:
                    count += 1
                else:
                    continue

            if count <= skip:
                continue

            yield {
                'id': img_id,
                'image': item['img'],
                'labels': labels,
            }

    def iterate_visual_item(self, skip=0, label=None, search=None):  
        for item in self.iterate(skip, label, search):
            record = {
                "tf_img_id": item['id'],
                "labels": item['labels'],
                "b64": embed_image_html(item['image'])
            }
            
            yield record
    
    @property
    def distributions(self):
        return self.reader.get_distributions()

    @property
    def labels(self):
        return self.classes

    @property
    def total(self):
        return self.reader.total_entries
