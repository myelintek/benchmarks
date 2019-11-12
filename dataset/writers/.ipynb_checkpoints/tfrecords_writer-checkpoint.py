import os
import six
import json
import click
import numpy as np
import tensorflow as tf
from PIL import Image


CLASSES_FILENAME = 'labels.txt'
CLASSES_JSON_FILENAME = 'classes.json'
IMAGE_PER_SHARD = 10000


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_image(path):
    with tf.gfile.GFile(path, 'rb') as f:
        image = f.read()
    return image


class ProgressHook(object):
    def __init__(self, total_steps, update_callback=None, split='train'):
        self._total_steps = total_steps
        self._current_step = 0
        self._update_callback = update_callback
        self._split = split

    @property
    def total_steps(self):
        return self._total_steps
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def update(self, step=1):
        self._current_step += step
        if self._update_callback:
            self._update_callback(self._current_step, self._total_steps, split=self._split)
        


class ClickProgressHook(ProgressHook):

    def __init__(self, total_steps, update_callback=None, split='train'):
        super(ClickProgressHook, self).__init__(total_steps, None, split=split)

    def __enter__(self):
        self.bar = click.progressbar(length=self.total_steps)
        return self

    def update(self, step=1):
        self.bar.update(step)


class TFRecordsWriter(object):
    
    def __init__(self, reader, output_dir, split='data', sharded=False, progress_callback=None, progress_hook=ClickProgressHook):
        self._reader = reader
        self._output_dir = output_dir
        self._split = split
        self._sharded = sharded
        self._progress_hook = progress_hook
        self._progress_callback = progress_callback

    def convert_to_example(self):
        pass
        
    def _save_labels(self, labels):
        classes_file = os.path.join(self._output_dir, CLASSES_FILENAME)
        classes_json_file = os.path.join(self._output_dir, CLASSES_JSON_FILENAME)
        with open(classes_file, 'w') as out:
            out.write('\n'.join(labels) + '\n')
        with open(classes_json_file, 'w') as jf:
            json.dump(labels, jf)
  
    def _save_sharded(self):
        total_records = self._reader.total
        num_shards = total_records // IMAGE_PER_SHARD
        if total_records % IMAGE_PER_SHARD or num_shards == 0:
            num_shards += 1
        print("--------------------------------save_sharded-------------------------\n")
        filepaths, lables, texts = self._reader.get_shuffle_paths()
        assert len(filepaths) == len(labels)
        assert len(filepaths) == len(texts)
        
        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        vspacing = np.linspace(0, len(filepaths), num_shards + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])
        shard = 0
        with click.progressbar(length=total_records) as bar:
            for enu, shard_range in enumerate(ranges):
                output_filename = '%s-%.5d-of-%.5d' % (self._split, enu, num_shards)
                output_file = os.path.join(self._output_dir, output_filename)
                writer = tf.python_io.TFRecordWriter(output_file)

                files_in_shard = np.arange(shard_range[0], shard_range[1], dtype=int)
                for i in files_in_shard:
                    filepath = filepaths[i]
                    label = labels[i]
                    text = texts[i]
                    image = read_image(filepath)

                    image_pil = Image.open(six.BytesIO(image))
                    width = image_pil.width
                    height = image_pil.height
                    example = self.convert_to_example({
                        'filepath': filepath,
                        'image': image,
                        'height': height,
                        'width': width,
                        'lable': label,
                        'text': text,
                    })

                    writer.write(example.SerializeToString())
                    bar.update(1)
                writer.close()
        print("--------------------------------save file list-------------------------\n")
        if self._split == 'train':
            with open(os.path.join(self._output_dir, 'train_list.txt'), 'w') as f:
                f.write('\n'.join(filepaths))
        else:
            with open(os.path.join(self._output_dir, 'validation_list.txt'), 'w') as f:
                f.write('\n'.join(filepaths))
    
    def _save_single(self):
        record_file = os.path.join(
            self._output_dir,
            '{}.tfrecords'.format(self._split)
        )
        if self._split == 'train':
            lwriter = open(os.path.join(self._output_dir, 'train_list.txt'), 'w')
        else:
            lwriter = open(os.path.join(self._output_dir, 'validation_list.txt'), 'w')
        writer = tf.python_io.TFRecordWriter(record_file)
        try:
            with self._progress_hook(self._reader.total, self._progress_callback, split=self._split) as bar:
                for record in self._reader.iterate():
                    example = self.convert_to_example(record)
                    writer.write(example.SerializeToString())
                    lwriter.write(record['filepath']+'\n')
                    bar.update(1)
        finally:
            writer.close()
            lwriter.close()
        
    def save(self):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._reader.parse()
            
        if self._sharded:
            self._save_sharded()
        else:
            self._save_single()

        labels = self._reader.labels()
        self._save_labels(labels)       
        
