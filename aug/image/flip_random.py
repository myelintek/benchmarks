import tensorflow as tf

def run(image, batch_position=0):
    distorted_image = tf.image.random_flip_left_right(image)
    return distorted_image
