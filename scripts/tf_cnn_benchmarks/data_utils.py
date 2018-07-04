# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""tf.data utility methods.

Collection of utility methods that make CNN benchmark code use tf.data easier.
"""
import tensorflow as tf

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile

def parse_example_proto(example_serialized):
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  return features['image/encoded'], label

def normalized_image(images):
  # Rescale from [0, 255] to [0, 2]
  images = tf.multiply(images, 1. / 127.5)
  # Rescale to [-1, 1]
  return tf.subtract(images, 1.0)

class DataLoader(object):
  def __init__(self,
    height, width, batch_size, num_splits,
    cpu_device, params,
    gpu_devices, data_type, dataset):

    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.num_splits = num_splits
    self.gpu_devices = gpu_devices
    self.dtype = data_type
    with tf.device(cpu_device):
      if params.eval:
        subset = 'validation'
      else:
        subset = 'train'

      self.function_buffering_resources = []
      remote_fn, args = minibatch_fn(
        height=height,
        width=width,
        batch_size=batch_size,
        num_splits=num_splits,
        preprocess_fn=self.parse_and_preprocess,
        dataset=dataset,
        subset=subset,
        train=(not params.eval),
        cache_data=params.cache_data,
        num_threads=params.datasets_num_private_threads)
      for device_num in range(len(gpu_devices)):
        with tf.device(gpu_devices[device_num]):
          buffer_resource_handle = prefetching_ops.function_buffering_resource(
            f=remote_fn,
            target_device=cpu_device,
            string_arg=args[0],
            buffer_size=params.datasets_prefetch_buffer_size,
            shared_name=None)
          self.function_buffering_resources.append(buffer_resource_handle)

  def parse_and_preprocess(self, value, batch_position):
    image_buffer, label_index = parse_example_proto(value)
    image = self.preprocess(image_buffer)
    return (label_index, image)

  def preprocess(self, image_buffer):
    """Preprocessing image_buffer as a function of its batch position."""
    with tf.name_scope('distort_image'):
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      distorted_image = tf.image.random_flip_left_right(image)
      distorted_image = tf.image.resize_images(
        distorted_image, [self.height, self.width],
        tf.image.ResizeMethod.BILINEAR,
        align_corners=False)
      distorted_image.set_shape([self.height, self.width, 3])
      normalized = normalized_image(distorted_image)
    return tf.cast(normalized, self.dtype)

  def get_images_and_labels(self, device_num, data_type):
    """Given a FunctionBufferingResource obtains images and labels from it."""
    function_buffering_resource = self.function_buffering_resources[device_num]
    return prefetching_ops.function_buffering_resource_get_next(
      function_buffer_resource=function_buffering_resource,
      output_types=[data_type, tf.int32])


def build_prefetch_image_processing(height, width, batch_size, num_splits,
                                    preprocess_fn, cpu_device, params,
                                    gpu_devices, data_type, dataset):
  """"Returns FunctionBufferingResources that do image pre(processing)."""
  with tf.device(cpu_device):
    if params.eval:
      subset = 'validation'
    else:
      subset = 'train'

    function_buffering_resources = []
    remote_fn, args = minibatch_fn(
        height=height,
        width=width,
        batch_size=batch_size,
        num_splits=num_splits,
        preprocess_fn=preprocess_fn,
        dataset=dataset,
        subset=subset,
        train=(not params.eval),
        cache_data=params.cache_data,
        num_threads=params.datasets_num_private_threads)
    for device_num in range(len(gpu_devices)):
      with tf.device(gpu_devices[device_num]):
        buffer_resource_handle = prefetching_ops.function_buffering_resource(
            f=remote_fn,
            #output_types=[data_type, tf.int32],
            target_device=cpu_device,
            string_arg=args[0],
            buffer_size=params.datasets_prefetch_buffer_size,
            shared_name=None)
        function_buffering_resources.append(buffer_resource_handle)
    return function_buffering_resources


def get_images_and_labels(function_buffering_resource, data_type):
  """Given a FunctionBufferingResource obtains images and labels from it."""
  return prefetching_ops.function_buffering_resource_get_next(
      function_buffer_resource=function_buffering_resource,
      output_types=[data_type, tf.int32])


def create_iterator(batch_size,
                    num_splits,
                    batch_size_per_split,
                    preprocess_fn,
                    dataset,
                    subset,
                    train,
                    cache_data,
                    num_threads=None):
  """Creates a dataset iterator for the benchmark."""
  glob_pattern = dataset.tf_record_pattern(subset)
  file_names = gfile.Glob(glob_pattern)
  if not file_names:
    raise ValueError('Found no files in --data_dir matching: {}'
                     .format(glob_pattern))
  ds = tf.data.TFRecordDataset.list_files(file_names)
  ds = ds.apply(
      interleave_ops.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=10))
  if cache_data:
    ds = ds.take(1).cache().repeat()
  counter = tf.data.Dataset.range(batch_size)
  counter = counter.repeat()
  ds = tf.data.Dataset.zip((ds, counter))
  ds = ds.prefetch(buffer_size=batch_size)
  if train:
    ds = ds.shuffle(buffer_size=10000)
  ds = ds.repeat()
  ds = ds.apply(
      batching.map_and_batch(
          map_func=preprocess_fn,
          batch_size=batch_size_per_split,
          num_parallel_batches=num_splits))
  ds = ds.prefetch(buffer_size=num_splits)
  if num_threads:
    ds = threadpool.override_threadpool(
        ds,
        threadpool.PrivateThreadPool(
            num_threads, display_name='input_pipeline_thread_pool'))
    ds_iterator = ds.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         ds_iterator.initializer)
  else:
    ds_iterator = ds.make_one_shot_iterator()
  return ds_iterator


def minibatch_fn(height, width, batch_size, num_splits, preprocess_fn, dataset,
                 subset, train, cache_data, num_threads):
  """Returns a function and list of args for the fn to create a minibatch."""
  batch_size_per_split = batch_size // num_splits
  with tf.name_scope('batch_processing'):
    ds_iterator = create_iterator(batch_size, num_splits, batch_size_per_split,
                                  preprocess_fn, dataset, subset, train,
                                  cache_data, num_threads)
    ds_iterator_string_handle = ds_iterator.string_handle()

    @function.Defun(tf.string)
    def _fn(h):
      depth = 3
      remote_iterator = tf.data.Iterator.from_string_handle(
          h, ds_iterator.output_types, ds_iterator.output_shapes)
      labels, images = remote_iterator.get_next()
      images = tf.reshape(
          images, shape=[batch_size_per_split, height, width, depth])
      labels = tf.reshape(labels, [batch_size_per_split])
      return images, labels

    return _fn, [ds_iterator_string_handle]
