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
"""Contains the TFOpLambda layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import
import textwrap
from keras import backend as K
from keras.engine.base_layer import Layer
import tensorflow.compat.v2 as tf
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name


class TFOpLambda(Layer):
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
    self.function = function
    self.symbol = (
        get_canonical_name_for_symbol(
            self.function, add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(
            self.function, api_name='keras', add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      # Generate a name.
      # TFOpLambda layers avoid already-observed names,
      # because users cannot easily control the generated names.
      # Without this avoidance, users would be more likely to run
      # into unavoidable duplicate layer name collisions.
      # (For standard layers users could just set `name` when creating the
      # layer to work around a collision, but they can't do that for
      # auto-generated layers)
      if self.symbol:
        name = 'tf.' + self.symbol
      else:
        name = self.function.__name__
      kwargs['name'] = K.unique_object_name(
          name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      return self._call_wrapper(*args, **kwargs)

    self.call = tf.__internal__.decorator.make_decorator(
        function, _call_wrapper)

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(TFOpLambda, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    # Warning on every invocation will be quite irksome in Eager mode.
    self._already_warned = False

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def _call_wrapper(self, *args, **kwargs):
    created_variables = []

    def _variable_creator(next_creator, **creator_kwargs):
      var = next_creator(**creator_kwargs)
      created_variables.append(var)
      return var

    with tf.GradientTape(watch_accessed_variables=True) as tape, \
        tf.variable_creator_scope(_variable_creator):
      # We explicitly drop `name` arguments here,
      # to guard against the case where an op explicitly has a
      # `name` passed (which is susceptible to producing
      # multiple ops w/ the same name when the layer is reused)
      kwargs.pop('name', None)
      result = self.function(*args, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    if not created_variables and not accessed_variables:
      # In the common case that a Lambda layer does not touch a Variable, we
      # don't want to incur the runtime cost of assembling any state used for
      # checking only to immediately discard it.
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent("""
          The following Variables were created within a Lambda layer ({name})
          but are not tracked by said layer:
          {variable_str}
          The layer cannot safely ensure proper Variable reuse across multiple
          calls, and consquently this behavior is disallowed for safety. Lambda
          layers are not well suited to stateful computation; instead, writing a
          subclassed Layer is the recommend way to define layers with
          Variables.""").format(
              name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(
          textwrap.dedent("""
          The following Variables were used a Lambda layer's call ({name}), but
          are not present in its tracked objects:
          {variable_str}
          It is possible that this is intended behavior, but it is more likely
          an omission. This is a strong indication that this layer should be
          formulated as a subclassed Layer rather than a Lambda layer.""")
          .format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    # This method will be overridden in a unit test to raise an error, because
    # self.assertWarns is not universally implemented.
    return tf_logging.warning(msg)

  def get_config(self):
    if not self.symbol:
      raise ValueError('This Keras op layer was generated from %s, a method '
                       'that is not an exposed in the TensorFlow API. This '
                       'may have happened if the method was explicitly '
                       'decorated to add dispatching support, and it was used '
                       'during Functional model construction. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.' %
                       self.function)
    config = {'function': self.symbol}

    base_config = super(TFOpLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config['function']
    function = get_symbol_from_name(symbol_name)
    if not function:
      raise ValueError('TF symbol `tf.%s` could not be found.' % symbol_name)

    config['function'] = function

    return cls(**config)
