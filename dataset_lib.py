"""Benchmark dataset utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import tensorflow as tf
import pandas as pd

from tensorflow.python.ops import data_flow_ops
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.platform import gfile

from preprocessing_lib import RecordInputImagePreprocessor
from preprocessing_lib import COCOPreprocessor
from preprocessing_lib import AdvPreprocessor
import const
import aug_dir


def get_data_augmentations():
    return aug_dir.aug_list


class Dataset(object):
    """Abstract class for cnn benchmarks dataset."""

    def __init__(self,
                 name,
                 data_dir=None,
                 queue_runner_required=False,
                 num_classes=None):
        self.name = name
        self.data_dir = data_dir
        self._queue_runner_required = queue_runner_required
        self._num_classes = num_classes

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def reader(self):
        return tf.TFRecordReader()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, val):
        self._num_classes = val

    def __str__(self):
        return self.name

    def queue_runner_required(self):
        return self._queue_runner_required

    def use_synthetic_gpu_inputs(self):
        return not self.data_dir


class UserDataset(Dataset):
    def __init__(self, data_dir, subset=None):
        if data_dir:
            name = os.path.basename(os.path.normpath(data_dir))
            self._labels_file = os.path.join(data_dir, const.LABEL_FILE)
            self.labels = [line for line in open(self._labels_file) if line.strip()]
            num_classes = len(self.labels)
            super(UserDataset, self).__init__(
                name, data_dir=data_dir, num_classes=num_classes)
        else:
            super(UserDataset, self).__init__(
                name='imagenet', num_classes=1000)
        self.subset = subset

    def labels(self):
        return self.labels

    def extract_filenames(self, input_list):
        (inputs, labels, filenames) = input_list
        return (inputs, labels), filenames

    def num_examples_per_epoch(self, subset='train'):
        if self.subset is not None:
            subset = self.subset
        if subset == 'train':
            if not self.data_dir:
                return const.IMAGENET_NUM_TRAIN_IMAGES
            file_list = os.path.join(self.data_dir, const.TRAIN_LIST_FILE)
            if not os.path.isfile(file_list):
                return 0
            total_items = sum(1 for line in open(file_list))
            return total_items
        elif subset == 'validation':
            if not self.data_dir:
                return const.IMAGENET_NUM_VAL_IMAGES
            file_list = os.path.join(self.data_dir, const.VAL_LIST_FILE)
            if not os.path.isfile(file_list):
                return 0
            total_items = sum(1 for line in open(file_list))
            return total_items
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def get_input_preprocessor(self, input_preprocessor='default'):
        assert not self.use_synthetic_gpu_inputs()
        if input_preprocessor == 'default':
            return RecordInputImagePreprocessor
        elif input_preprocessor == 'coco':
            return COCOPreprocessor

    def tf_record_pattern(self, subset):
        if self.subset is not None:
            subset = self.subset
        if subset == 'train':
            return os.path.join(self.data_dir, '%s-*-of-*' % subset)
        elif subset == 'validation':
            return os.path.join(self.data_dir, '%s-*-of-*' % subset)
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


class AdvancedDataset(Dataset):
    def __init__(self, data_dir, subset=None):
        if data_dir:
            name = os.path.basename(os.path.normpath(data_dir))
            # no labels file for AdvancedDataset
            self.labels = [0]
            num_classes = len(self.labels)
            super(AdvancedDataset, self).__init__(
                name, data_dir=data_dir, num_classes=num_classes)
        else:
            super(AdvancedDataset, self).__init__(
                name='imagenet', num_classes=1000)
        self.subset = subset

    def num_examples_per_epoch(self, subset='train'):
        if self.subset is not None:
            subset = self.subset
        if subset == 'train':
            if not self.data_dir:
                return const.IMAGENET_NUM_TRAIN_IMAGES
            csv_file = os.path.join(self.data_dir, const.TRAIN_CSV_FILE)
            if os.path.isfile(csv_file):
                train_data = pd.read_csv(csv_file)
                return len(train_data.values)
            return 0
        elif subset == 'validation':
            if not self.data_dir:
                return const.IMAGENET_NUM_VAL_IMAGES
            csv_file = os.path.join(self.data_dir, const.VAL_CSV_FILE)
            if os.path.isfile(csv_file):
                val_data = pd.read_csv(csv_file)
                return len(val_data.values)
            return 0
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def get_input_preprocessor(self, input_preprocessor='default'):
        assert not self.use_synthetic_gpu_inputs()
        return AdvPreprocessor
