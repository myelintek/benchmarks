
from image_classification import ClassiReader
READERS = {
    'classification': ClassiReader,
}


class InvalidDataDirectory(Exception):
    """
    Error raised when the chosen intput directory for the dataset is not valid.
    """


def get_reader(reader):
    reader = reader.lower()
    if reader not in READERS:
        raise ValueError('"{}" is not a valid reader'.format(reader))
    return READERS[reader]
