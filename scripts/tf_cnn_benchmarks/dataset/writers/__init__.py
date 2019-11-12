from .classification_writer import ClassificationWriter # noqa
from .tfrecords_writer import ProgressHook

WRITERS = {
    'classification': ClassificationWriter,
}


def get_writer(writer):
    writer = writer.lower()
    if writer not in WRITERS:
        raise ValueError('"{}" is not a valid writer'.format(writer))
    return WRITERS[writer]
