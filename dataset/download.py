import os
from myelindl.log import logger
from myelindl.core.downloader import Downloader
from myelindl.core.dataset import bucket


class DSDownloader(Downloader):
    def after_run(self):
        self.output_log.close()
        self.progress = 1.
        # register to bucket!
        bucket_name = os.path.basename(self.download_path)
        bucket.add(self.username, bucket_name)
        logger.info("done.")
