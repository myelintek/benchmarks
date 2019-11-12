import os
import json
from .tfrecords_writer import ProgressHook

class EmptyWriter(object):
    def save():
        pass


class OnlyLabelsWriter(object):
    def __init__(self, reader, 
            output_dir, split='data', 
            progress_callback=None, progress_hook=ProgressHook):
        self.reader = reader
        self.output_dir = output_dir
        self._progress_hook = progress_hook
        self._progress_callback = progress_callback

    def save(self):
        # iterate to check format
        with self._progress_hook(self.reader.total, self._progress_callback, split='data') as updater:
            for items in self.reader.iterate():
                updater.update(1)

        if hasattr(self.reader, 'labels'):
            with open(os.path.join(self.output_dir, 'classes.json'), 'w') as f:
                json.dump(self.reader.labels, f)
