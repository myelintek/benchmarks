
import tensorflow as tf
import logging
import re

import sys
gcdir="./memory_saving_methods/openai_gradient_checkpointing"
sys.path.insert(0, gcdir)
import time
from memory_saving_methods.openai_gradient_checkpointing import memory_saving_gradients
from memory_saving_methods.openai_gradient_checkpointing import mem_util

# automatic checkpoint selectionion
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
  return memory_saving_gradients.gradients(ys, xs, grad_ys,
checkpoints='memory', **kwargs)

def gradients_speed(ys, xs, grad_ys=None, **kwargs):
  return memory_saving_gradients.gradients(ys, xs, grad_ys,
checkpoints='speed', **kwargs)



class Memory_Saving(object):

  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.memory_saving_method = benchmark_cnn.params.memory_saving_method
    self.param_server_device = benchmark_cnn.cpu_device
    #if (benchmark_cnn.variable_update != 'cpu_recomputing'):
    #   raise ValueError('variable_update needs to use cpu_recomputing, '\
    #                    'if memory_saving_method set to \'recomputing\'')

    # hack the gradients function call in tf
    tf.__dict__["gradients"] = gradients_memory
    logging.info("memory_saving: enable method = {}".format(self.memory_saving_method))


# 2018/10/26 deprecated, this class was dropped after integrate the variable_mgr into benchmark_cnn
# This class will be erased after 2018/12/10.
# class CPU_Recomputing_Updater(object):
# 
#   def __init__(self, raw_devices, param_server_device):
#     # TODO check whether the param_server_device is cpu or not
#     self.param_server_device = param_server_device
#     self.raw_devices = raw_devices
#     self.param_server_devices = self.get_ps_devices()
#     logging.info("CPU_Recomputing_Updater: setup parameter_server_device to {}, "\
#                  "param_server_devices = {}".format(
#                  self.param_server_device, self.param_server_devices))
# 
#   def get_ps_devices(self):
#     # ref: origin is in variable_mgr.py L212~L225
#     return [
#             tf.train.replica_device_setter(
#               worker_device=d,
#               ps_device=self.param_server_device,
#               ps_tasks=1) for d in self.raw_devices
#            ]
# 
#   def __replace_variable_prefix_name(self, sgv):
#     """replace the variable name to tower_0 from tower_i,
#        because the variables used to bind with grads get from
#        tf.trainable_variables() without filtering in original
#        version, so the variable name shoule be the same as 
#        tower_0 in the case of cpu parameter server.
#        Hence, we replace the tower_i and return to check 
#        whether the variable name correct or not.
#        logically, we want to average the grad of the same
#        variable across towers, this replace is correct in our 
#        case.
#     """
#     # replace the tower_i with tower_0
#     pattern = re.compile(r"tower_\d+", re.IGNORECASE)
#     return re.sub(pattern, r'tower_0', sgv.name)
# 
#   def aggregate_gradients_using_copy_with_variable_colocation(
#       self, tower_grads, use_mean, check_inf_nan):
#     """Aggregate gradients, colocating computation with the gradient's variable.
#     Args:
#       tower_grads: List of lists of (gradient, variable) tuples. The outer list
#         is over towers. The inner list is over individual gradients. All variables
#         of the same gradient across towers must be the same (that is,
#         tower_grads[x][a][1] == tower_grads[y][a][1] for all indices x, y, and a)
#       use_mean: if True, mean is taken, else sum of gradients is taken.
#       check_inf_nan: If true, check grads for nans and infs.
#     Returns:
#       The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
#         gradient has been averaged across all towers. The variable is chosen from
#         the first tower. The has_nan_or_inf indicates the grads has nan or inf.
#     """
#     agg_grads = []
#     has_nan_or_inf_list = []
# 
#     for single_grads in zip(*tower_grads):
#       # Note that each single_grads looks like the following:
#       #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#       # tensor: [0][0] ; variable [0][1]
#       var = single_grads[0][1]
# 
#       logging.debug("var = {}, single_grads = {}".format(var, single_grads))
#       for _, v in single_grads:
#         #assert v == var
#         assert var.name == self.__replace_variable_prefix_name(v)
# 
#       with tf.device(var.device):
#         grad_and_var, has_nan_or_inf = self.aggregate_single_gradient_using_copy(
#             single_grads, use_mean, check_inf_nan)
#         # append to list
#         agg_grads.append(grad_and_var)
#         has_nan_or_inf_list.append(has_nan_or_inf)
# 
#     if check_inf_nan:
#       return agg_grads, tf.reduce_any(has_nan_or_inf_list)
#     else:
#       return agg_grads, None
# 
#   def aggregate_single_gradient_using_copy(self, grad_and_vars, use_mean,
#                                            check_inf_nan):
#     """Calculate the average gradient for a shared variable across all towers.
#     Note that this function provides a synchronization point across all towers.
#     Args:
#       grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
#         (gradient, variable) pair within the outer list represents the gradient
#         of the variable calculated for a single tower, and the number of pairs
#         equals the number of towers.
#       use_mean: if True, mean is taken, else sum of gradients is taken.
#       check_inf_nan: check grads for nans and infs.
#     Returns:
#       The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
#         gradient has been averaged across all towers. The variable is chosen from
#         the first tower. The has_nan_or_inf indicates the grads has nan or inf.
#     """
#     from tensorflow.python.ops import gradients_impl
# 
#     grads = [g for g, _ in grad_and_vars]
#     if any(isinstance(g, tf.IndexedSlices) for g in grads):
#       # TODO(reedwm): All-reduce IndexedSlices more effectively.
#       grad = gradients_impl._AggregateIndexedSlicesGradients(grads)  # pylint: disable=protected-access
#     else:
#       grad = tf.add_n(grads)
# 
#     if use_mean and len(grads) > 1:
#       grad = tf.scalar_mul(1.0 / len(grads), grad)
# 
#     v = grad_and_vars[0][1]
#     if check_inf_nan:
#       with tf.name_scope('check_for_inf_and_nan'):
#         has_nan_or_inf = tf.logical_not(tf.reduce_all(tf.is_finite(grads)))
#       return (grad, v), has_nan_or_inf
#     else:
#       logging.debug("calculated (grad = {}, v = {})".format(grad, v))
#       return (grad, v), None
# 
#   def get_gradients_to_apply(self, gradient_state):
#     all_grads = []
#     device_grads = gradient_state
#     agg_grads, grad_has_inf_nan = (
#         self.aggregate_gradients_using_copy_with_variable_colocation(
#             device_grads,
#             use_mean=True,
#             check_inf_nan=False)) #enable_auto_loss_scale: false in myelintek case
#     all_grads.append(agg_grads)
#     return all_grads


