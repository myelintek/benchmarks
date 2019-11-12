# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

#from absl import app
from absl import flags as absl_flags
import tensorflow as tf
import flags

flags.DEFINE_string('network_dir', 'networks', 'network file path.')
flags.DEFINE_string('network', 'network.py', 'network file name')
flags.DEFINE_integer('small_chunk', 1, 'accumulate gradients.')
flags.DEFINE_string('memory_saving_method', None,
                    'setup the memory saving method, 1. recomputing 2. TBD ')
flags.DEFINE_enum('lr_policy', 'multistep', ('multistep', 'exp'), 'learning_rate policy')
flags.DEFINE_boolean('aug_flip', True, 'whether randomly flip left or right dataset')
flags.DEFINE_integer('stop_accu_epoch', 0, 'early stop when accuracy does not increase 1% for'
                     'numbers of epochs')
flags.DEFINE_boolean('save_stop', True, 'whether to save checkpoint when killing process')
flags.DEFINE_list('aug_list', [], 'Specify a list of augmentation function names to apply '
                  'during training.')

import benchmark_cnn
import memory_saving as ms
import benchmark_handler
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)


def main(positional_arguments):
    # Command-line arguments like '--distortions False' are equivalent to
    # '--distortions=True False', where False is a positional argument. To prevent
    # this from silently running with distortions, we do not allow positional
    # arguments.
    assert len(positional_arguments) >= 1
    if len(positional_arguments) > 1:
        raise ValueError('Received unknown positional arguments: %s'
                         % positional_arguments[1:])

    params = benchmark_cnn.make_params_from_flags()
    handler = benchmark_handler.Handler(params)
    params = handler.params
    params = benchmark_cnn.setup(params)
    bench = benchmark_cnn.BenchmarkCNN(
        params,
        dataset = handler.dataset,
        model = handler.model)
    handler.set_bench(bench)
    if getattr(bench.input_preprocessor, 'set_aug_list', None):
        bench.input_preprocessor.set_aug_list(params.aug_list)
    bench.benchmark_one_step = handler.benchmark_one_step
    bench.print_eval_results = handler.print_eval_results
    bench.check_early_stop = handler.check_early_stop

    bench.accum_grads = handler.accum_grads
    bench.build_fetches_forward = handler.build_fetches_forward
    if params.memory_saving_method == 'recomputing':
        bench.memory_saving = ms.Memory_Saving(benchmark_cnn=bench)
#    tfversion = util.tensorflow_version_tuple()
#    logging.info('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

    bench.print_info()
    bench.run()


if __name__ == '__main__':
    # avoid to use app.run, which will change logging format
    tf.app.run(main)
