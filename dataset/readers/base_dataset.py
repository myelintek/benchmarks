import os
import tensorflow as tf
import sonnet as snt

from luminoth.datasets.exceptions import InvalidDataDirectory


class BaseDataset(snt.AbstractModule):
    def __init__(self, config, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        self._dataset_dir = config.dataset.dir
        self._num_epochs = config.train.num_epochs
        self._batch_size = config.train.batch_size
        self._split = config.dataset.split
        self._random_shuffle = config.train.random_shuffle
        self._seed = config.train.seed

        self._fixed_resize = (
            'fixed_height' in config.dataset.image_preprocessing and
            'fixed_width' in config.dataset.image_preprocessing
        )
        if self._fixed_resize:
            self._image_fixed_height = (
                config.dataset.image_preprocessing.fixed_height
            )
            self._image_fixed_width = (
                config.dataset.image_preprocessing.fixed_width
            )

    def _build(self):
        # Find split file from which we are going to read.
        split_path = os.path.join(
            self._dataset_dir, '{}.tfrecords'.format(self._split)
        )
        if not tf.gfile.Exists(split_path):
            raise InvalidDataDirectory(
                '"{}" does not exist.'.format(split_path)
            )
        filenames = tf.gfile.Glob(split_path)
        ds = tf.data.TFRecordDataset(filenames)
        ds = ds.map(self.read_record)
        #ds = ds.batch(self._batch_size)
        ds = ds.repeat() # repeat the input indefinitely
        iterator = ds.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        element = iterator.get_next()
        return element
