from __future__ import print_function
import os
import sys
import re
import tensorflow as tf
from collections import namedtuple
from absl import flags as absl_flags
# from tf benchmark
import flags
import shutil
import numpy as np
import socket


class CheckpointNotFoundException(Exception):
  pass


def tensorflow_version_tuple():
  v = tf.__version__
  major, minor, patch = v.split('.')
  return (int(major), int(minor), patch)


def get_global_step_ckpt(ckpt_dir):
  model_checkpoint_path = _get_checkpoint_to_load(ckpt_dir)
  global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]
  if not global_step.isdigit():
    global_step = 0
  else:
    global_step = int(global_step)
  return global_step


def _get_checkpoint_to_load(ckpt_dir):
  """Returns which checkpoint to load.

  Args:
    ckpt_dir: Path to a folder of checkpoints or full path to a checkpoint.

  Returns:
    Full path to checkpoint to load.

  Raises:
    CheckpointNotFoundException: If checkpoint is not found.
  """
  p = re.compile(r'ckpt-\d+$')
  if p.search(ckpt_dir):
    model_checkpoint_path = ckpt_dir
  else:
    # Finds latest checkpoint in directory provided
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      model_checkpoint_path = ckpt.model_checkpoint_path
    else:
      raise CheckpointNotFoundException('No checkpoint file found in dir:{}'.
                                        format(ckpt_dir))
  return model_checkpoint_path


def params_eval(params):
  params = params._replace(eval=True, num_eval_epochs=1,
                           variable_update='replicated',
                           all_reduce_spec='nccl')
  return params


Params = namedtuple('Params', flags.param_specs.keys())  # pylint: disable=invalid-name
def make_params_from_flags():
  flag_values = {name: getattr(absl_flags.FLAGS, name)
                 for name in flags.param_specs.keys()}
  return Params(**flag_values)


def get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter, scale=1):
  if scale == 1:
    # TODO(laigd): rename 'images' to maybe 'inputs', same below.
    return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))
  else:
    return 'images/sec: %.1f' % speed_mean


def get_perf_timing(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
  speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
  return speed_mean, speed_uncertainty, speed_jitter


def copytree(src, dst, symlinks=False, ignore=None):
  for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
      shutil.copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)


def get_folder_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP
