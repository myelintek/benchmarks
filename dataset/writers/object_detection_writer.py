import os
import json
from luminoth.tools.dataset.writers.object_detection_writer import ObjectDetectionWriter
from .tfrecords_writer import ProgressHook


class MLSteamObjectDetectionWriter(ObjectDetectionWriter):

    def __init__(self, reader, output_dir, split='data',
            progress_callback=None, progress_hook=ProgressHook, **kwargs):
        super(MLSteamObjectDetectionWriter, self).__init__(
            reader, output_dir, split, **kwargs)
        self._progress_hook = progress_hook(self._reader.total, progress_callback)

        self._classes = self._reader.get_classes()

    def _record_to_tf(self, record):
        self._progress_hook.update(1)
        return super(MLSteamObjectDetectionWriter, self)._record_to_tf(record)
    
