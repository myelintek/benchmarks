#!/usr/bin/env python2

import argparse
from collections import Counter
import logging
import math
import os
import Queue
import random
import re
import shutil
import sys
import threading
import time
from datetime import datetime
import six

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import numpy as np
import PIL.Image
import json
# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import multiprocessing
import tensorflow as tf


IMAGE_PER_SHARD = 10000
TRAIN_LIST_FILE = 'train_list.txt'
VAL_LIST_FILE = 'validation_list.txt'
TRAIN_DB = 'train_db'
VAL_DB = 'val_db'
LABELS_FILE = 'labels.txt'


logger = logging.getLogger('create_db')


class Error(Exception):
    pass


class BadInputFileError(Error):
    """Input file is empty"""
    pass


class ParseLineError(Error):
    """Failed to parse a line in the input file"""
    pass


class LoadError(Error):
    """Failed to load image[s]"""
    pass


class WriteError(Error):
    """Failed to write image[s]"""
    pass


class Hdf5DatasetExtendError(Error):
    """Failed to extend an hdf5 dataset"""
    pass


class DbWriter(object):
    """
    Abstract class for writing to databases
    """

    def __init__(self, output_dir, image_height, image_width, image_channels):
        self._dir = output_dir
        os.makedirs(output_dir)
        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._count = 0

    def write_batch(self, batch):
        raise NotImplementedError

    def count(self):
        return self._count

    
def _find_datadir_labels(input_file, labels_file):
    """
    Search for subdirection under train or validation foler as labels name
    input_file:
        /imagenet/train/n01440764/n01440764_8834.JPEG 0
        /imagenet/train/n15075141/n15075141_45683.JPEG 999
    labels:
        [n01440764, n15075141]
    """
    labelfile = []
    with open(labels_file) as lfile:
        for line in lfile:
            labelfile.append(line.rstrip())
    filenames = []
    labels = []
    texts = []
    category_counters = {}
    with open(input_file) as infile:
        for line in infile:
            match = re.match(r'(.+)\s+(\d+)\s*$', line)
            if match is None:
                raise ParseLineError
            if len(match.groups()) == 1:
              label = 0
            filepath = match.group(1)
            label = match.group(2)
            filenames.append(filepath)
            labels.append(int(label))
            texts.append(labelfile[int(label)])
            if label in category_counters:
                category_counters[label] += 1
            else:
                category_counters[label] = 1

    for key in category_counters:
        logger.debug('Category {} has {} images.'.format(key, category_counters[key]))

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]

    return filenames, labels, texts


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        # force use CPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        self._resize_size = tf.placeholder(dtype=tf.int32)
        self._resize_jpeg_data = tf.placeholder(dtype=tf.string)
        self._resize_jpeg_decoded = tf.image.decode_jpeg(self._resize_jpeg_data, channels=3)
        self._resize_jpeg_cropped = tf.cast(tf.image.resize_images(
                                            self._resize_jpeg_decoded,
                                            [self._resize_size, self._resize_size]), tf.uint8)
        self._resize_jpeg = tf.image.encode_jpeg(self._resize_jpeg_cropped)

        self._resize_png_data = tf.placeholder(dtype=tf.string)
        self._resize_png_decoded = tf.image.decode_png(self._resize_png_data, channels=3)
        self._resize_png_cropped = tf.cast(tf.image.resize_images(
                                           self._resize_png_decoded,
                                           [self._resize_size, self._resize_size]), tf.uint8)
        self._resize_png = tf.image.encode_png(self._resize_png_cropped)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def resize_jpeg(self, image_data, size):
        image = self._sess.run(self._resize_jpeg,
            feed_dict={self._resize_jpeg_data: image_data, self._resize_size: size})
        return image

    def resize_png(self, image_data, size):
        image = self._sess.run(self._resize_png,
            feed_dict={self._resize_png_data: image_data, self._resize_size: size})
        return image

def _is_png(filename):
    return filename.endswith('.png')


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    image_jpeg = None
    if _is_png(filename):
        image_jpeg = coder.png_to_jpeg(image_data)
    else:
        image_jpeg = image_data
    image = coder.decode_jpeg(image_jpeg)
    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    return image_jpeg, height, width


def _convert_to_example(filename, image_buffer, height, width, label, text, bbox=[]):
    """Build an Example proto for an example.
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    colorspace = 'RGB'
    image_format = 'JPEG'
    channels = 3
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([label] * len(xmin)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


def _create_example_from_example(example,**kwargs):

    filename = tf.compat.as_str_any(example.features.feature['image/filename'].bytes_list.value[0])

    height = int(example.features.feature['image/height'].int64_list.value[0])
    width = int(example.features.feature['image/width'].int64_list.value[0])
    label = int(example.features.feature['image/class/label'].int64_list.value[0])
    text = tf.compat.as_text(example.features.feature['image/class/text'].bytes_list.value[0])
    image_buffer = example.features.feature['image/encoded'].bytes_list.value[0]

    if 'filename' in kwargs:
        filename = kwargs.pop('filename', '')

    if 'height' in kwargs:
        height = kwargs.pop('height', 0)

    if 'width' in kwargs:
        width = kwargs.pop('width', 0)

    if 'label' in kwargs:
        label = kwargs.pop('label', 0)

    if 'text' in kwargs:
        text = kwargs.pop('text', '')

    if 'image_buffer' in kwargs:
        image_buffer = kwargs.pop('image_buffer', '')

    return _convert_to_example(filename, image_buffer, height, width, label, text)


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
        labels, texts, num_shards, output_dir, image_count, bboxes):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]
            bbox = []
            if bboxes:
                bbox = bboxes[i]
            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                logger.warning(e)
                logger.warning('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue
            example = _convert_to_example(filename, image_buffer, height, width, label, text, bbox)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            image_count[thread_index] = counter
            if not counter % 1000:
                logger.info('%s [thread %d]: Processed %d of %d images in thread batch.' %
                            (datetime.now(), thread_index, counter, num_files_in_thread))
        writer.close()
        logger.info('%s [thread %d]: Wrote %d images to %s' %
                    (datetime.now(), thread_index, shard_counter, output_file))
        shard_counter = 0
        logger.info('%s [thread %d]: Wrote %d images to %d shards.' %
                    (datetime.now(), thread_index, counter, num_files_in_thread))


def _calculate_num_shard(total_size):
    num_shards = total_size // IMAGE_PER_SHARD
    if num_shards % 2 or num_shards == 0:
        num_shards += 1 # make this number even or one

    return num_shards


def _find_image_bounding_boxes(filenames, image_to_bboxes):
    """Find the bounding boxes for a given image file.
    Args:
      filenames: list of strings; each string is a path to an image file.
      image_to_bboxes: dictionary mapping image file names to a list of
        bounding boxes. This list contains 0+ bounding boxes.
    Returns:
      List of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    """
    num_image_bbox = 0
    bboxes = []
    for f in filenames:
        basename = os.path.basename(f)
        if basename in image_to_bboxes:
            bboxes.append(image_to_bboxes[basename])
            num_image_bbox += 1
        else:
            bboxes.append([])
    print('Found %d images with bboxes out of %d images' % (
          num_image_bbox, len(filenames)))
    return bboxes


def _build_bounding_box_lookup(bounding_box_file):
    """Build a lookup from image file to bounding boxes.
    Args:
      bounding_box_file: string, path to file with bounding boxes annotations.
        Assumes each line of the file looks like:
          n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
        where each line corresponds to one bounding box annotation associated
        with an image. Each line can be parsed as:
          <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
        Note that there might exist mulitple bounding box annotations associated
        with an image file. This file is the output of process_bounding_boxes.py.
    Returns:
      Dictionary mapping image file names to a list of bounding boxes. This list
      contains 0+ bounding boxes.
    """
    lines = tf.gfile.GFile(bounding_box_file, 'r').readlines()
    images_to_bboxes = {}
    num_bbox = 0
    num_image = 0
    for l in lines:
        if l:
            parts = l.split(',')
            assert len(parts) == 5, ('Failed to parse: %s' % l)
            filename = parts[0]
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])
            box = [xmin, ymin, xmax, ymax]
            if filename not in images_to_bboxes:
                images_to_bboxes[filename] = []
                num_image += 1
            images_to_bboxes[filename].append(box)
            num_bbox += 1
    print('Successfully read %d bounding boxes '
          'across %d images.' % (num_bbox, num_image))
    return images_to_bboxes


def create_tfrecords_db(input_file, output_dir, labels_file, prefix, bbox_file):
    """ find labels and convert to tfrecords
    """
    if os.path.exists(output_dir):
        logger.info('ouput folder exist: {}'.format(output_dir))
    else:
        os.makedirs(output_dir)

    filenames, labels, texts = _find_datadir_labels(input_file, labels_file)
    bboxes = []
    if os.path.exists(bbox_file):
        image_to_bboxes = _build_bounding_box_lookup(bbox_file)
        bboxes = _find_image_bounding_boxes(filenames, image_to_bboxes)
    assert len(filenames) == len(labels)
    assert len(filenames) == len(texts)
    if bboxes:
        assert len(filenames) == len(bboxes)

    num_shards = _calculate_num_shard(len(filenames))

    if num_shards < (multiprocessing.cpu_count() // 2):
      num_threads = num_shards
    else:
      num_threads = (multiprocessing.cpu_count() // 2)
      while num_shards % num_threads:
        num_threads -= 1

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    logger.info('Launching %d threads for spacings: %s' % (num_threads, ranges))

    coord = tf.train.Coordinator()
    coder = ImageCoder()
    threads = []
    image_count = [0] * num_threads
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, prefix, filenames,
                labels, texts, num_shards, output_dir, image_count, bboxes) 
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    wait_time = time.time()
    while sum(image_count) < len(filenames):
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (sum(image_count), len(filenames)))
            wait_time = time.time()
        time.sleep(0.2)

    # Wait for all the threads to terminate.
    coord.join(threads)
    if 'train' in prefix:
        listfile = open(os.path.join(output_dir, TRAIN_LIST_FILE), 'w')
    else:
        listfile = open(os.path.join(output_dir, VAL_LIST_FILE), 'w')
    for filename in filenames:
        listfile.write("%s\n" % filename)
    logger.info('%s images written to database' % len(filenames))


def _dataset_iterator(ds, sess):
    iterator = ds.make_one_shot_iterator()
    next_row = iterator.get_next()

    try:
        while True:
            yield sess.run(next_row)
    except tf.errors.OutOfRangeError:
        pass


def _copy_file(source, target):
    if os.path.exists(source):
        with open(source, 'r') as source_file, open(target, 'w') as target_file:
            target_file.writelines(source_file.readlines())


def _merge_changelog(changelogs):

    change_log_map = {}

    for log in changelogs:

        if log['id'] not in change_log_map:
            change_log_map[log['id']] = {}

        if 'delete' in change_log_map[log['id']]:
            continue

        if log['operation'] == 'delete':
            change_log_map[log['id']]['delete'] = {}

        elif log['operation'] == 'relabel':
            change_log_map[log['id']]['relabel'] = log['label']

        elif log['operation'] == 'duplicate':
            change_log_map[log['id']]['duplicate'] = log['number']

    item_number_change = 0
    for key in change_log_map:
        if 'delete' in change_log_map[key]:
            item_number_change -= 1
        if 'duplicate' in change_log_map[key]:
            item_number_change += int(change_log_map[key]['duplicate'])

    return change_log_map, item_number_change


def create_tfrecords_db_from_db(dataset_dir, db_name, prefix, changelog_file, output_dir):

    #
    # Copy files
    #

    # copy label file
    _copy_file(os.path.join(dataset_dir, LABELS_FILE),
               os.path.join(output_dir, LABELS_FILE))

    labels = []
    label_ids = {}
    id = 0
    with open(os.path.join(output_dir, LABELS_FILE), 'r') as label_file:
        for line in label_file.readlines():
            label = line.strip()
            labels.append(label)
            label_ids[label] = id
            id += 1

    if db_name == TRAIN_DB:
        # copy train_db_file
        _copy_file(os.path.join(dataset_dir, TRAIN_FILE),
                   os.path.join(output_dir, TRAIN_FILE))

    # copy val_db file
    if db_name ==  VAL_DB:
        _copy_file(os.path.join(dataset_dir, VAL_FILE),
                   os.path.join(output_dir, VAL_FILE))
    db_file_pattern = '%s-*-of-*' % os.path.join(dataset_dir, prefix)
    logger.info('Search TFrecords in %s' % db_file_pattern)
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(db_file_pattern))

    changelogs = {}
    item_num_change = 0
    if os.path.exists(changelog_file):
        logger.info('Changelog Exist %s' % changelog_file)
        with open(changelog_file, 'r') as cf:
            changelogs, item_num_change = _merge_changelog(json.load(cf))

        logger.info('changelog datat {}'.format(changelogs))
    else:
        logger.info('Changelog Not Exist %s' % changelog_file)

    total_record = 0
    with tf.Session() as sess:
        for row in _dataset_iterator(dataset, sess):
            total_record += 1

    total_record += item_num_change
    num_shards = _calculate_num_shard(total_record)
    category_counters = {}
    record_id = 0
    record_count = 0
    if db_name == TRAIN_DB:
        list_file = os.path.join(output_dir, TRAIN_LIST_FILE)
    else:
        list_file = os.path.join(output_dir, VAL_LIST_FILE)
    with tf.Session() as sess, open(list_file, 'w') as lf:
        writer = None
        for row in _dataset_iterator(dataset, sess):
            if (record_count % IMAGE_PER_SHARD) == 0:
                if writer:
                    writer.close()

                shard = record_count // IMAGE_PER_SHARD
                output_filename = '%s-%.5d-of-%.5d' % (prefix, shard, num_shards)
                writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, output_filename))

            example = tf.train.Example()
            example.ParseFromString(row)

            label_id = int(example.features.feature['image/class/label'].int64_list.value[0])
            if label_id in category_counters:
                category_counters[label_id] += 1
            else:
                category_counters[label_id] = 1

            if (record_count % 1000) == 0:
                logger.debug('Processed %d/%d' % (record_count, total_record))

            if record_id not in changelogs:
                writer.write(example.SerializeToString())
                filename = tf.compat.as_str_any(example.features.feature['image/filename'].bytes_list.value[0])
                lf.write("%s\n" % filename)
                record_id += 1
                record_count += 1
                continue

            changelog = changelogs[record_id]
            record_id += 1

            if 'delete' in changelog:
                logger.info("Delete data {}".format(record_id))
                category_counters[label_id] -= 1
                continue

            if 'relabel' in changelog:
                label_text = changelog['relabel']
                if label_text not in label_ids:
                    logger.error("Wrong label, label not found in label file: %s" % label_text)
                else:
                    logger.info("Relabel data {} to {}".format(record_id, label_text))
                    category_counters[label_id] -= 1
                    label_id = label_ids[label_text]
                    example = _create_example_from_example(example, label=label_id, text=label_text)

            if 'duplicate' in changelog:
                example_str = example.SerializeToString();
                filename = tf.compat.as_str_any(example.features.feature['image/filename'].bytes_list.value[0])
                logger.info("Duplicate data {} x{}".format(record_id, changelog['duplicate']))
                for i in xrange(int(changelog['duplicate'])):
                    writer.write(example_str)
                    lf.write("%s\n" % filename)
                    record_count += 1
                    category_counters[label_id] += 1
            else:
                writer.write(example.SerializeToString())
                filename = tf.compat.as_str_any(example.features.feature['image/filename'].bytes_list.value[0])
                lf.write("%s\n" % filename)
                record_count += 1
                category_counters[label_id] += 1

        if writer:
            writer.close()

        for key in category_counters:
            logger.debug('Category {} has {} images.'.format(key, category_counters[key]))
        logger.info('%s images written to database' % record_count)
        

def _create_tfrecords(image_count, write_queue, batch_size, output_dir,
                      summary_queue, num_threads,
                      mean_files=None,
                      encoding=None,
                      lmdb_map_size=None,
                      **kwargs):
    """
    Creates the TFRecords database(s)
    """
    LIST_FILENAME = 'list.txt'

    if not tf:
        raise ValueError("Can't create TFRecords as support for Tensorflow "
                         "is not enabled.")

    wait_time = time.time()
    threads_done = 0
    images_loaded = 0
    images_written = 0
    image_sum = None
    compute_mean = bool(mean_files)

    os.makedirs(output_dir)

    # We need shards to achieve good mixing properties because TFRecords
    # is a sequential/streaming reader, and has no random access.

    num_shards = 2 if image_count < 100000 else 128

    writers = []
    with open(os.path.join(output_dir, LIST_FILENAME), 'w') as outfile:
        for shard_id in xrange(num_shards):
            shard_name = 'SHARD_%03d.tfrecords' % (shard_id)
            filename = os.path.join(output_dir, shard_name)
            writers.append(tf.python_io.TFRecordWriter(filename))
            outfile.write('%s\n' % (filename))

    shard_id = 0
    while (threads_done < num_threads) or not write_queue.empty():

        # Send update every 2 seconds
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (images_written, image_count))
            wait_time = time.time()

        processed_something = False

        if not summary_queue.empty():
            result_count, result_sum = summary_queue.get()
            images_loaded += result_count
            # Update total_image_sum
            if compute_mean and result_count > 0 and result_sum is not None:
                if image_sum is None:
                    image_sum = result_sum
                else:
                    image_sum += result_sum
            threads_done += 1
            processed_something = True

        if not write_queue.empty():
            writers[shard_id].write(write_queue.get())
            shard_id += 1
            if shard_id >= num_shards:
                shard_id = 0
            images_written += 1
            processed_something = True

        if not processed_something:
            time.sleep(0.2)

    if images_loaded == 0:
        raise LoadError('no images loaded from input file')
    logger.debug('%s images loaded' % images_loaded)

    if images_written == 0:
        raise WriteError('no images written to database')
    logger.info('%s images written to database' % images_written)

    for writer in writers:
        writer.close()

        
def _fill_load_queue(filename, queue, shuffle):
    """
    Fill the queue with data from the input file
    Print the category distribution
    Returns the number of lines added to the queue

    NOTE: This can be slow on a large input file, but we need the total image
        count in order to report the progress, so we might as well read it all
    """
    total_lines = 0
    valid_lines = 0
    distribution = Counter()

    with open(filename) as infile:
        if shuffle:
            lines = infile.readlines()  # less memory efficient
            random.shuffle(lines)
            for line in lines:
                total_lines += 1
                try:
                    result = _parse_line(line, distribution)
                    valid_lines += 1
                    queue.put(result)
                except ParseLineError:
                    pass
        else:
            for line in infile:  # more memory efficient
                total_lines += 1
                try:
                    result = _parse_line(line, distribution)
                    valid_lines += 1
                    queue.put(result)
                except ParseLineError:
                    pass

    logger.debug('%s total lines in file' % total_lines)
    if valid_lines == 0:
        raise BadInputFileError('No valid lines in input file')
    logger.info('%s valid lines in file' % valid_lines)

    for key in sorted(distribution):
        logger.debug('Category %s has %d images.' % (key, distribution[key]))

    return valid_lines


def _parse_line(line, distribution):
    """
    Parse a line in the input file into (path, label)
    """
    line = line.strip()
    if not line:
        raise ParseLineError

    # Expect format - [/]path/to/file.jpg 123
    match = re.match(r'(.+)\s+(\d+)\s*$', line)
    if match is None:
        raise ParseLineError

    path = match.group(1)
    label = int(match.group(2))

    distribution[label] += 1

    return path, label


def _calculate_batch_size(image_count, is_hdf5=False, hdf5_dset_limit=None,
                          image_channels=None, image_height=None, image_width=None):
    """
    Calculates an appropriate batch size for creating this database
    """
    if is_hdf5 and hdf5_dset_limit is not None:
        return min(100, image_count, hdf5_dset_limit / (image_channels * image_height * image_width))
    else:
        return min(100, image_count)


def _calculate_num_threads(batch_size, shuffle):
    """
    Calculates an appropriate number of threads for creating this database
    """
    if shuffle:
        return min(10, int(round(math.sqrt(batch_size))))
    else:
        # XXX This is the only way to preserve order for now
        # This obviously hurts performance considerably
        return 1


def _initial_image_sum(width, height, channels):
    """
    Returns an array of zeros that will be used to store the accumulated sum of images
    """
    if channels == 1:
        return np.zeros((height, width), np.float64)
    else:
        return np.zeros((height, width, channels), np.float64)


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _array_to_tf_feature(image, label, encoding):
    """
    Creates a tensorflow Example from a numpy.ndarray
    if not encoding:
        image_raw = image.tostring()
        encoding_id = 0
    else:
        s = StringIO()
        if encoding == 'png':
            PIL.Image.fromarray(image).save(s, format='PNG')
            encoding_id = 1
        elif encoding == 'jpg':
            PIL.Image.fromarray(image).save(s, format='JPEG', quality=90)
            encoding_id = 2
        else:
            raise ValueError('Invalid encoding type')
        image_raw = s.getvalue()
    """
    encoding_id = 0
    depth = image.shape[2] if len(image.shape) > 2 else 1

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(image.shape[0]),
                'width': _int64_feature(image.shape[1]),
                'depth': _int64_feature(depth),
                'label': _int64_feature(label),
                #'image_raw': _bytes_feature(image_raw),
                'image_raw': _float_array_feature(image.flatten()),
                'encoding':  _int64_feature(encoding_id),
                # @TODO(tzaman) - add bitdepth flag?
            }
        ))
    return example.SerializeToString()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-Db tool - DIGITS')

    # Positional arguments

    parser.add_argument('input_file',
                        help='An input file of labeled images')
    parser.add_argument('output_dir',
                        help='Path to the output database')
    parser.add_argument('--labels_file',
                        help='Path to the label file')
    parser.add_argument('--prefix', default='shard',
                        help='prefix for the output database')
    parser.add_argument('--parent_dataset_folder', default='',
                        help='specify parent dataset folder to apply change log')
    parser.add_argument('--changelog', default='',
                        help='specify changlog file paht tot laod changlog')

    parser.add_argument('--db_name', default='',
                        help='specify db_name to apply change log')
    parser.add_argument('--bounding_box_file',
                        default='imagenet_2012_bounding_boxes.csv',
                        help='Bounding box csv file')

    args = vars(parser.parse_args())

    try:
        logger.info("parent_dataset_folder:%s,changelog:%s"  % (args['parent_dataset_folder'], args['changelog']))
        if args['parent_dataset_folder'] != '':
            create_tfrecords_db_from_db(args['parent_dataset_folder'],
                                        args['db_name'], args['prefix'],
                                        args['changelog'],
                                        args['output_dir'])
        else:
            create_tfrecords_db(args['input_file'], args['output_dir'],
                                args['labels_file'], args['prefix'],
                                args['bounding_box_file'])

    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
