import click
import tensorflow as tf

from .readers import get_reader, READERS
from .writers import get_writer


@click.command()
@click.option('dataset_type', '--type', type=click.Choice(READERS.keys()), required=True)  # noqa
@click.option('--data-dir', required=True, help='Where to locate the original data.')  # noqa
@click.option('--output-dir', required=True, help='Where to save the transformed data.')  # noqa
@click.option('splits', '--split', required=True, multiple=True, help='The splits to transform (ie. train, test, val).')  # noqa
@click.option('--debug', is_flag=True, help='Set level logging to DEBUG.')
def conversion(dataset_type, data_dir,  output_dir, splits, debug):
    tf.logging.set_verbosity(tf.logging.INFO)
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    try:
        reader = get_reader(dataset_type)
    except ValueError as e:
        tf.logging.error('Error getting reader: {}'.format(e))
        return

    try:
        writer = get_writer(dataset_type)
        for split in splits:
            split_reader = reader(data_dir, split)
            writer = writer(split_reader, output_dir, split)
            writer.save()

            tf.logging.info('Composition per class ({})'.format(split))
    except ValueError as e:
        tf.logging.error('Error reading dataset: {}'.format(e))
