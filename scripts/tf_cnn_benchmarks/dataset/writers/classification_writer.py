import tensorflow as tf
from .tfrecords_writer import (
    TFRecordsWriter,
    _int64_feature,
    _float_feature,
    _bytes_feature,
    _float_array_feature,
)


def _convert_to_example(filename, image_buffer, height, width, label, text):
    """Build an Example proto for an example.
    """
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class ClassificationWriter(TFRecordsWriter):
    def __init__(self, reader, output_dir, split='data',**kwargs):
        super(ClassificationWriter, self).__init__(reader, output_dir, split, **kwargs)

    def convert_to_example(self, record):
        return _convert_to_example(
            record['filepath'],
            record['image_buffer'],
            record['height'],
            record['width'],
            record['label'],
            record['text'],
        )
