# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the SlicingOpLambda class."""
from keras.engine import keras_tensor
from keras.layers.core.tf_op_lambda import TFOpLambda
import tensorflow.compat.v2 as tf


class SlicingOpLambda(TFOpLambda):
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    super(SlicingOpLambda, self).__init__(function, **kwargs)

    original_call = self.call

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      # Turn any slice dicts in the args back into `slice` objects.
      # This conversion cannot use nest.flatten/map_structure,
      # because dicts are flattened by nest while slices aren't.
      # So, map_structure would only see the individual elements in the
      # dict.
      # This can't use map_structure_up_to either because the 'shallowness' of
      # the shallow tree would have to vary depending on if only one dim or
      # multiple are being sliced.
      new_args = []
      for arg in args:
        arg = _dict_to_slice(arg)
        if isinstance(arg, (list, tuple)):
          new_arg = []
          for sub_arg in arg:
            new_arg.append(_dict_to_slice(sub_arg))
          arg = new_arg
        new_args.append(arg)

      # Handle the kwargs too.
      new_kwargs = {}
      for key, value in kwargs.items():
        value = _dict_to_slice(value)
        if isinstance(value, (list, tuple)):
          new_value = []
          for v in value:
            new_value.append(_dict_to_slice(v))
          value = new_value
        new_kwargs[key] = value

      return original_call(*new_args, **new_kwargs)

    self.call = tf.__internal__.decorator.make_decorator(
        original_call, _call_wrapper)


def _slice_to_dict(x):
  if isinstance(x, slice):
    return {'start': x.start, 'stop': x.stop, 'step': x.step}
  return x


def _dict_to_slice(x):
  if isinstance(x, dict):
    return slice(x['start'], x['stop'], x['step'])
  return x


class TFSlicingOpDispatcher(tf.__internal__.dispatch.OpDispatcher):
  """A global dispatcher that allows building a functional model with TF Ops."""

  def __init__(self, op):
    self.op = op

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    args = tf.nest.map_structure(_slice_to_dict, args)
    kwargs = tf.nest.map_structure(_slice_to_dict, kwargs)
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in tf.nest.flatten([args, kwargs])):
      return SlicingOpLambda(self.op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED


for slicing_op in [
    tf.__operators__.getitem,  # pylint: disable=protected-access
    tf.compat.v1.boolean_mask,
    tf.boolean_mask,
    tf.__operators__.ragged_getitem
]:
  TFSlicingOpDispatcher(slicing_op).register(slicing_op)
