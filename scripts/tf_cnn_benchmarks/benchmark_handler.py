"""
  core functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import os
import time
import json

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import constants
from tensorflow.python.ops import data_flow_ops
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
import logging

from dataset_lib import UserDataset
from sighdl import SignalHandler
import util
import model_lib as model

# How many digits to show for the loss and accuracies during training.
LOSS_AND_ACCURACY_DIGITS_TO_SHOW = 3
DEBUG = False

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
  """Returns a piecewise learning rate tensor.

  Args:
    piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
      parameter
    global_step: Scalar tensor representing the global step.
    num_batches_per_epoch: float indicating the number of batches per epoch.

  Returns:
    A scalar float tensor, representing the learning rate.

  Raises:
    ValueError: piecewise_learning_rate_schedule is not formatted correctly.
  """
  pieces = piecewise_learning_rate_schedule.split(';')
  if len(pieces) % 2 == 0:
    raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                     'number of components')
  values = []
  boundaries = []
  for i, piece in enumerate(pieces):
    if i % 2 == 0:
      try:
        values.append(float(piece))
      except ValueError:
        raise ValueError('Invalid learning rate: ' + piece)
    else:
      try:
        boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
      except ValueError:
        raise ValueError('Invalid epoch: ' + piece)
  return tf.train.piecewise_constant(global_step, boundaries, values,
                                     name='piecewise_learning_rate')



def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
  """Returns a learning rate tensor based on global_step.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    global_step: Scalar tensor representing the global step.
    num_examples_per_epoch: The number of examples per epoch.
    model: The model.Model object to obtain the default learning rate from if no
      learning rate is specified.
    batch_size: Number of examples per step

  Returns:
    A scalar float tensor, representing the learning rate. When evaluated, the
    learning rate depends on the current value of global_step.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  with tf.name_scope('learning_rate'):
    num_batches_per_epoch = (float(num_examples_per_epoch) / batch_size)

    if params.piecewise_learning_rate_schedule:
      if (params.init_learning_rate is not None or
          params.learning_rate_decay_factor or
          params.minimum_learning_rate or params.num_epochs_per_decay):
        raise ValueError('No other learning rate-related flags can be '
                         'specified if --piecewise_learning_rate_schedule is '
                         'specified')
      learning_rate = get_piecewise_learning_rate(
          params.piecewise_learning_rate_schedule,
          global_step, num_batches_per_epoch)
    elif params.init_learning_rate is not None:
      learning_rate = params.init_learning_rate
      if (params.num_epochs_per_decay > 0 and
          params.learning_rate_decay_factor > 0):
        decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(
            params.init_learning_rate,
            global_step,
            decay_steps,
            params.learning_rate_decay_factor,
            staircase=True)

        if params.minimum_learning_rate != 0.:
          learning_rate = tf.maximum(learning_rate,
                                     params.minimum_learning_rate)
    else:
      learning_rate = model.get_learning_rate(global_step, batch_size)
    if params.num_learning_rate_warmup_epochs > 0 and (
        params.init_learning_rate is not None or
        params.piecewise_learning_rate_schedule):
      warmup_steps = int(num_batches_per_epoch *
                         params.num_learning_rate_warmup_epochs)
      init_lr = params.init_learning_rate
      if init_lr is None:
        init_lr = float(params.piecewise_learning_rate_schedule.split(';')[0])
      warmup_lr = init_lr * tf.cast(global_step, tf.float32) / tf.cast(
          warmup_steps, tf.float32)
      learning_rate = tf.cond(global_step < warmup_steps,
                              lambda: warmup_lr, lambda: learning_rate)

  return learning_rate


class Handler(object):
    def __init__(self, params):
        params = params._replace(#num_learning_rate_warmup_epochs=5,
                                 variable_update='replicated',
                                 all_reduce_spec='nccl',
                                 print_training_accuracy=True)
        if not params.eval and not params.forward_only:
            params = params._replace(eval_during_training_every_n_epochs=1.,
                                     num_eval_epochs=1., summary_verbosity=1)
        self.batch_size = params.batch_size * params.num_gpus
        self.task_index = 0
        self.cpu_device = '/cpu:0'
        self.dataset = UserDataset(params.data_dir)
        subset = 'validation' if params.eval else 'train'
        self.num_examples_per_epoch = self.dataset.num_examples_per_epoch(subset)
        print("num_examples_per_epoch: {}".format(self.num_examples_per_epoch))
        self.num_steps_per_epoch = int(float(self.num_examples_per_epoch)/self.batch_size)
        if self.num_steps_per_epoch < 1:
            raise ValueError('batch size {} is larger than total number of dataset {}'.format(self.batch_size, self.num_examples_per_epoch))
        self.num_val_examples_per_epoch = self.dataset.num_examples_per_epoch('validation')
        if self.num_val_examples_per_epoch < self.batch_size:
            raise ValueError('batch size {} is larger than total number of  validation dataset {}'.format(self.batch_size, self.num_val_examples_per_epoch))
        display_every = self.num_steps_per_epoch // 10
        if display_every == 0:
            display_every = 1
        if display_every > 100:
            display_every = 100
        if not params.eval:
            params = params._replace(display_every=display_every,
                                     save_summaries_steps=self.num_steps_per_epoch)
        if params.small_chunk > 1:
             params = params._replace(variable_update='replicated',
                                      all_reduce_spec='nccl')
        if params.use_fp16:
            params = params._replace(fp16_enable_auto_loss_scale=True,
                                     all_reduce_spec=None)
        if params.num_gpus > 8:
             params = params._replace(all_reduce_spec=None)
        self.global_step = 0
        if params.forward_only:
            self.global_step = util.get_global_step_ckpt(params.train_dir)
            logging.info('self.gobal_step: {}'.format(self.global_step))
            params = params._replace(num_warmup_batches=0)
        self.params = params 
        self.model = None
        if self.params.network_dir is not None:
            self.model = model.get_model(self.params.network_dir,
                                         self.params.network,
                                         params=params)
        self.print_graph = True
        self.local_step = 0
        self.local_step_diff = 0
        self.val_accuracy_history = []
        self.signal = None
        if not params.eval and not params.forward_only and params.save_stop:
            self.signal = SignalHandler()

    def set_bench(self, bench):
        self.task_index = bench.task_index
        self.cpu_device = bench.cpu_device
        self.model = bench.model
        self.num_workers = bench.num_workers
        if self.global_step > 0:
            bench.num_batches += self.global_step
        #if self.params.forward_only:
        #    bench.num_batches = self.global_step + 1

    def print_results(self, results, step):
        if step < 0 and self.local_step == 0 and self.local_step_diff == 0:
            self.local_step_diff = self.local_step - step
        if (step >= 0 and (step == 0 or
            (step + 1) % self.params.display_every == 0)):
            epoch = (float(step) / self.num_steps_per_epoch)
            log_str = 'Training (epoch %.2f):' % (epoch)
            if 'average_loss' in results:
                lossval = results['average_loss']
                log_str += ' loss = %.3f,' % (lossval)
            if 'top_1_accuracy' in results:
                accuracy = results['top_1_accuracy']
                log_str += ' accuracy = %.4f,' % (accuracy)
            if 'learning_rate' in results:
                learning_rate = results['learning_rate']
                log_str += ' learning_rate = %.5f,' % (learning_rate)
            logging.info(log_str)

    def print_eval_results(self, results):
        log_str = 'Validation (epoch %.2f):' % (float(self.local_step - self.local_step_diff) / self.num_steps_per_epoch)
        for key, value in results.iteritems():
            log_str += ' %s = %.4f,' % (key, value)
            if key == 'accuracy':
                self.val_accuracy_history.append(value)
        logging.info(log_str)
        if hasattr(self.model, 'softmax') and self.model.softmax and self.params.eval:
            logging.info('Predictions for image 0: ' + json.dumps(self.model.softmax))

    def check_early_stop(self):
        if self.signal is not None and self.signal.kill_now:
            return True
        if (self.params.stop_accu_epoch > 0 and
                len(self.val_accuracy_history) > self.params.stop_accu_epoch):
            cmp_epoch = -1 - self.params.stop_accu_epoch
            delta = self.val_accuracy_history[-1]-self.val_accuracy_history[cmp_epoch]
            logging.debug('comparing accuracies {} and {}'.format(
                          self.val_accuracy_history[cmp_epoch],
                          self.val_accuracy_history[-1]))
            if delta < 0.01:
                print('Early stop for accuracy {} and {}'.format(
                             self.val_accuracy_history[cmp_epoch],
                             self.val_accuracy_history[-1]))
                return True
        return False

    def _create_var(self, var, name):
        with tf.colocate_with(var):
            return tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False, name=name)

    def _zero_var(self, var):
        with tf.colocate_with(var):
            return var.assign(tf.zeros_like(var))

    def _assign_add(self, var, grad):
        with tf.colocate_with(var):
            return var.assign_add(tf.divide(grad, self.params.small_chunk))
            #return var.assign_add(grad)

    def accum_grads(self, grads, fp32_params):
        if DEBUG:
            #print_grads = tf.print("grads: ",grads[0][0][0])
            grads[0] = tf.Print(grads[0], [grads[0]], "this is grads")
        #with tf.device(self.cpu_device):
        local_step = tf.Variable(0, name="local_step", trainable=False, dtype=tf.int32)
        with tf.colocate_with(local_step):
            local_step_inc = tf.assign_add(local_step, 1)
        if self.print_graph:
            for p in fp32_params:
                shape = p.get_shape()
                logging.info("{} \t{} \t{}".format(p.op.name, shape.as_list(), shape.num_elements()))
            self.print_graph = False
        with tf.control_dependencies(None):
            accum_grads = [self._create_var(tv, "accum_grads") for tv in fp32_params]
        def zero_func():
            zero_ops = [self._zero_var(tv) for tv in accum_grads]
            return tf.group(*zero_ops)
        # zero gradients every small chunk begin
        cond_zero_ops = tf.cond(tf.equal((local_step % self.params.small_chunk), 0),
                                zero_func, tf.no_op, name="cond_zero_ops")
        with tf.control_dependencies([cond_zero_ops]):
            with tf.control_dependencies([local_step_inc]):
                accum_grads = [self._assign_add(a, g) for a, g in zip(accum_grads, grads)]
        if DEBUG:
            accum_grads[0] = tf.Print(accum_grads[0], [accum_grads[0]], "this is accum_grads")
        grads = accum_grads
        return grads

    def build_fetches_forward(self, global_step, all_logits, losses, device_grads,
                              enqueue_ops, update_ops, all_accuracy_ops, phase_train):
        if not phase_train:
            return
        fetches_forward = {}
        for name, ops in all_accuracy_ops.items():
            # For fetches that starts with 'tensor:', keep dimension and skip reducing
            # them to scalars.
            if name.startswith(constants.UNREDUCED_ACCURACY_OP_PREFIX):
                fetches_forward[name[len(constants.UNREDUCED_ACCURACY_OP_PREFIX):]] = ops[0]
            else:
                fetches_forward[name] = tf.reduce_sum(ops) / self.batch_size
                if self.task_index == 0 and self.params.summary_verbosity >= 1:
                    tf.summary.scalar(name, fetches_forward[name])
        # TODO(reedwm): Greatly simplify the learning rate code.
        if (self.params.variable_update == 'horovod' or
            self.params.variable_update == 'collective_all_reduce'):
            # Each worker independently increments global_step.
            examples_per_step = self.batch_size * self.num_workers
        else:
            # global_step is shared by all workers, and so every iteration
            # global_step is incremented by num_workers.
            examples_per_step = self.batch_size
        with tf.device(self.cpu_device):
            #print_global_step = tf.print("this is global_step: ", global_step)
            #with tf.control_dependencies([print_global_step]):
            learning_rate = get_learning_rate(self.params, global_step,
                              self.num_examples_per_epoch,
                              self.model, examples_per_step)
            with tf.name_scope('average_loss'):
                average_loss = tf.reduce_mean(losses)
        with tf.device(self.cpu_device):
            if self.task_index == 0 and self.params.summary_verbosity >= 1:
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar(self.params.loss_type_to_report, average_loss)
        fetches_forward['average_loss'] = average_loss
        fetches_forward['learning_rate'] = learning_rate
        grads = [[g for g, _ in grad_vars] for grad_vars in device_grads]
        #print_grads = tf.print("this is forward grads: ", grads[0][0][0])
        #with tf.control_dependencies([print_grads]):
        #    accum_op = tf.group(*grads)
        accum_op = tf.group(*grads)
        fetches_forward['accum_grads'] = accum_op
        # increase global step
        fetches_forward_list = nest.flatten(list(fetches_forward.values()))
        main_fetch_forward_group = tf.group(*fetches_forward_list, name='main_fetch_forward_group')
        with tf.device(self.cpu_device), tf.name_scope('inc_global_step'):
            with tf.control_dependencies([main_fetch_forward_group]):
                fetches_forward['inc_global_step'] = global_step.assign_add(1)
        self.fetches_forward = fetches_forward

    def benchmark_one_step(self, sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       partitioned_graph_file_prefix,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       benchmark_logger=None,
                       collective_graph_key=0):
        """Advance one step of benchmarking."""
        run_options = None
        run_metadata = None
        summary_str = None
        start_time = time.time()
        # step before zero accum_grads
        if DEBUG:
            print("local_step: {}".format(self.local_step))
        if (self.local_step+1) % self.params.small_chunk == 0:
            if DEBUG:
                print("run fetches")
            if summary_op is None:
                results = sess.run(fetches, options=run_options,
                                   run_metadata=run_metadata)
            else:
                (results, summary_str) = sess.run([fetches, summary_op],
                                                  options=run_options,
                                                  run_metadata=run_metadata)
        else:
            if DEBUG:
                print("run fetches_forward")
            results = sess.run(self.fetches_forward, options=run_options,
                               run_metadata=run_metadata)
        if not params.forward_only:
            lossval = results['average_loss']
        else:
            lossval = 0.
        if image_producer is not None:
            image_producer.notify_image_consumption()
        train_time = time.time() - start_time
        step_train_times.append(train_time)
        if (show_images_per_sec and step >= 0 and
            (step == 0 or (step + 1) % params.display_every == 0)):
            speed_mean, speed_uncertainty, speed_jitter = util.get_perf_timing(
                batch_size, step_train_times)
            log_str = '%i\t%s\t%.*f' % (
                step + 1,
                util.get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter),
                LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
            logging.info(log_str)
        if benchmark_logger:
            benchmark_logger.log_metric(
                'current_examples_per_sec', speed_mean, global_step=step + 1)
            if 'top_1_accuracy' in results:
                benchmark_logger.log_metric(
                    'top_1_accuracy', results['top_1_accuracy'], global_step=step + 1)
                benchmark_logger.log_metric(
                    'top_5_accuracy', results['top_5_accuracy'], global_step=step + 1)
        self.print_results(results, step)
        self.local_step += 1
        return (summary_str, lossval)
