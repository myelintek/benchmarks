#
# https://www.tensorflow.org/tutorials/estimators/cnn#building_the_cnn_mnist_classifier
import model_lib as model
import tensorflow as tf
from tensorflow import layers


class UserModel(model.NNModel):
  def __init__(self, params=None):
    super(UserModel, self).__init__('Lenet5TF', model.MODEL_TYPE_IMAGE_CLASSIFICATION, 28, params=params)

  def add_inference(self, inputs, training, nclass):
    df = 'channels_first' if self.data_format == 'NCHW' else 'channels_last'
    conv1 = layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], padding="same", data_format=df, activation=tf.nn.relu)
    pool1 = layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, data_format=df)
    conv2 = layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", data_format=df, activation=tf.nn.relu)
    pool2 = layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, data_format=df)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = layers.dropout(inputs=dense, rate=0.4, training=training)
    logits = layers.dense(inputs=dropout, units=nclass)
    return logits
