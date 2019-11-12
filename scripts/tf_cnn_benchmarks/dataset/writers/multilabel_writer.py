import tensorflow as tf
from .tfrecords_writer import (
    TFRecordsWriter,
    _int64_feature,
    _int64_list_feature,
    _float_feature,
    _bytes_feature,
    _float_array_feature,
    ClickProgressHook,
)

def _convert_to_example(filename, image_buffer, height, width, label, text):
    """Build an Example proto for an example.
    """
    colorspace = 'L'
    channels = 1
    image_format = 'PNG'


    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_list_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class MultilabelWriter(TFRecordsWriter):
    def __init__(self, reader, output_dir, split='data', progress_callback=None,  progress_hook=ClickProgressHook):
        super(MultilabelWriter, self).__init__(reader, output_dir, split, False, progress_callback,  progress_hook)

    def convert_to_example(self, record):
        return _convert_to_example(
            record['filepath'],
            record['image'],
            record['height'],
            record['width'],
            record['labels_encoded'],
            record['labels_str'],
        )
