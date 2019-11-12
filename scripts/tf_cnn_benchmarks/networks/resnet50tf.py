"""
  https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
"""
import model_lib as model
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training, data_format, zero_init=False):
  if zero_init:
    return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, gamma_initializer=tf.zeros_initializer(), training=training, fused=True)
  else:
    return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

##
## ResNet blocks
##
def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  shortcut = inputs
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, zero_init=True)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs

def block_layer(inputs, filters, bottleneck, blocks, strides,
                training, name, data_format):
  layer_name = name + '_1'
  with tf.variable_scope(layer_name):
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
      return conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides,
                                  data_format)

  for i in range(1, blocks):
    layer_name = name + '_{}'.format(i+1)
    with tf.variable_scope(layer_name):
      inputs = _bottleneck_block_v1(inputs, filters, training, None, 1, data_format)

  return inputs


## ResNet50
class UserModel(model.NNModel):
  def __init__(self, params=None):
    super(UserModel, self).__init__('ResNet50TF', model.MODEL_TYPE_IMAGE_CLASSIFICATION, 224, params=params)

    self.resnet_size = 50
    self.bottleneck =True
    self.num_filters=64
    self.kernel_size=7
    self.conv_stride=2
    self.first_pool_size=3
    self.first_pool_stride=2
    self.block_sizes=[3, 4, 6, 3]
    self.block_strides=[1, 2, 2, 2]
    
  def add_inference(self, inputs, training, nclass):
    df = 'channels_first' if self.data_format == 'NCHW' else 'channels_last'
    inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=df)

    inputs = batch_norm(inputs, training, df)
    inputs = tf.nn.relu(inputs)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=self.first_pool_size,
        strides=self.first_pool_stride, padding='SAME',
        data_format=df)
    for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            blocks=num_blocks, strides=self.block_strides[i],
            training=training, name='block_layer{}'.format(i + 1),
            data_format=df)
    axes = [2, 3] if df == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    inputs = tf.squeeze(inputs, axes)
    inputs = tf.layers.dense(inputs=inputs, units=nclass)
    return inputs
