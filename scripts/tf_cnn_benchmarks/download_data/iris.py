
import gzip
import os
import struct

import numpy as np
import PIL.Image

from downloader import DataDownloader


class IrisDownloader(DataDownloader):
    """
    See details about the Iris dataset here:
    """

    def urlList(self):
        return [
            'http://download.tensorflow.org/data/iris_training.csv',
            'http://download.tensorflow.org/data/iris_test.csv',
        ]

    def uncompressData(self):
        pass

    def processData(self):
        pass
