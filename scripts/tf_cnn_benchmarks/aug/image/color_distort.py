import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.python.layers import utils


def distort_color(image, batch_position=0, distort_color_in_yiq=False):
    with tf.name_scope('distort_color'):
        def distort_fn_0(image=image):
            """Variant 0 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            if distort_color_in_yiq:
                image = distort_image_ops.random_hsv_in_yiq(
                    image, lower_saturation=0.5, upper_saturation=1.5,
                    max_delta_hue=0.2 * math.pi)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image

        def distort_fn_1(image=image):
            """Variant 1 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            if distort_color_in_yiq:
                image = distort_image_ops.random_hsv_in_yiq(
                    image, lower_saturation=0.5, upper_saturation=1.5,
                    max_delta_hue=0.2 * math.pi)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            return image

        image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                                 distort_fn_1)
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

def run(image, batch_position=0):
    distorted_image = tf.cast(image, dtype=tf.float32)
    # Images values are expected to be in [0,1] for color distortion.
    distorted_image /= 255.
    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, batch_position)
    # Note: This ensures the scaling matches the output of eval_image
    distorted_image *= 255
    return distorted_image
