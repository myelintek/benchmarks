import os
import time
import shutil
import json
import time
import traceback
import subprocess
import signal

import myelindl.utils as utils
from myelindl.log import logger
from myelindl.core.run import ERROR, Run, DONE, INIT
from bucket import get_dataset_path, add_tag, remove_tag
from .visualize import reader_dataset_items

SUPPORTED_CONVERSION = {
    'classification':['classification'],
    'object_detection': ['kitti', 'coco', 'yolo'],
    'mlultilable': ['chestxray'],
    'lidar_npy' : ['lidar_npy'],
    'segmentation': ['segmentation']
    #  'csv': ['csv'],
}


CONVERSION_CATEGORIES = {
    'classification'     : 'classification',
    'kitti'     : 'object_detection',
    'coco'      : 'object_detection',
    'chestxray' : 'multilabel',
    'csv'       : 'csv',
    'lidar_npy' : 'lidar_npy',
    'yolo': 'object_detection',
    'segmentation': 'segmentation'
}

SKIP_CONVERT = ['lidar_npy']

class ConversionTask(Run):
    def __init__(self, username, dataset_id, input_dir, reader_type, parameters={}):
        
        logger.debug('Convert inint {} '.format(reader_type))
        id = 'conv-{}-{}-{}'.format(username, reader_type, time.time())
        name = id
        super(ConversionTask, self).__init__(id, username, name, persistent=False) 
        self.dataset_id = dataset_id
        self.input_dir = input_dir if input_dir else get_dataset_path(self.dataset_id)
        self.reader_type = reader_type
        self.output_dir = self._get_output_dir()
        self.parameters = parameters
        self.metadata = {}
        self.error_message = []
        self.skip = reader_type in SKIP_CONVERT
        logger.debug('Convert init finish {} '.format(self.reader_type))
        self.status = INIT

    def _dict_cust(self, d):
        if '_finish_callback' in d:
            del d['_finish_callback']
        if 'reader' in d:
            del d['reader']
        if 'writer' in d:
            del d['writer']
        if 'p' in d:
            del d['p']
        return d
    
    def offer_resources(self, resources):
        key = 'convert_dataset_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    def _get_output_dir(self):
        output_dir = os.path.join(self.input_dir, 'converted', self.reader_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def on_status_update(self, new_status, time):
        if not self.dataset_id:
            return
        status = {'status': new_status}
        if self.error_message:
            status['err_msg'] = ' '.join(self.error_message)

        self.metadata.update(status)
        add_tag(self.dataset_id, {
            self.reader_type: self.metadata
        })

    def _task_arguments(self):
        return [
            'python',
            '/build/myelindl/tools/convert_dataset.py',
            '--username={}'.format(self.username),
            '--dataset_id={}'.format(self.dataset_id),
            '--input_dir={}'.format(self.input_dir),
            '--reader_type={}'.format(self.reader_type),
            '--parameters={}'.format(json.dumps(self.parameters)),
        ]

    def progress_update(self, current, total, split):
        current = int(current)
        total = float(total)
    
        if current % 100 == 0 or current == total:
            self.progress= current  / total
            self.metadata.update({'progress': self.progress})
            add_tag(self.dataset_id, {
                self.reader_type: self.metadata
            })


    def process_output(self, message):
        if not message:
            return False
        
        if message.startswith('Progress'):
            [current, total, split] = message.split("=")[1].split('/')
            self.progress_update(current, total, split)
            return True
        elif message.startswith('Metadata'):
            [split, total_entries] = message.split("=")[1].split(':')
            self.metadata[split] = {'total_entries':total_entries}
            return True
        elif message.startswith('Error'):
            self.error_message.append(message.split("=")[1])
            return True
        else:
            return False

    def run(self, resources):
        if self.skip:
            self.metadata.update({'status': 'Done'})
            self.metadata.update({'data':0})
            add_tag(self.dataset_id, {
                self.reader_type: self.metadata,
            })
            self.status = DONE
            return
            
        args = self._task_arguments()
        self.p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        
        unknown_output = []
        self.before_run()
        try:
            while self.p.poll() is None:
                for line in utils.nonblocking_readlines(self.p.stdout):
                    if self.aborted.is_set():
                        self.p.send_signal(signal.SIGKILL) 
                    if line is not None:
                        self.output_log.write(line)
                        line = line.strip()
                    if line:
                        if not self.process_output(line):
                            unknown_output.append(line)  
                        else:
                            time.sleep(0.05)
                    time.sleep(0.01) # do not remove this line. 
        except Exception as e:
            self.p.terminate()
            self.status = ERROR
            logger.warning('Convert dataset {} to {} fail. {}'.format(self.dataset_id, self.reader_type, str(e)))
            logger.debug(traceback.format_exc(e))
            logger.debug(unknown_output)
            raise
        finally:
            self.after_run()

        if self.p.returncode != 0:
            self.status = ERROR
            logger.warning('Convert dataset {} to {} fail. return code {}, {}'.format(self.dataset_id, self.reader_type, self.p.returncode, self.error_message))
            logger.debug(unknown_output)
        else:
            self.status = DONE

def get_supported_conversion():
    return SUPPORTED_CONVERSION


def get_converted_path(bucket_id, format):
    return os.path.join(
        get_dataset_path(bucket_id),
        'converted',
        format
    )

        
def get_converted_labels(id, format):
    """get_dataset_labels

    :param id: published dataset id
    :returns: the labels of this dataset
    """
    labels=[]
    with open(os.path.join(get_converted_path(id,format), 'classes.json')) as f:
        labels = json.load(f)

    return labels


def create(username, bucket_id, reader_type, parameters):
    return ConversionTask(
        username=username,
        dataset_id=bucket_id,
        input_dir=get_dataset_path(bucket_id),
        reader_type=reader_type,
        parameters=parameters,
    )


def delete(bucket_id, format): 
    path = get_dataset_path(bucket_id)
    if not os.path.exists(path):
        status = ERROR
        raise ValueError('Target Bucket not exists {}'.format(bucket_id))

    remove_tag(bucket_id, format)
    
    output_dir = get_converted_path(bucket_id, format)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
 
