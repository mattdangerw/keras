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
"""Contains the ClassMethod helper."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend as K
from keras.engine.base_layer import Layer
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name


class ClassMethod(Layer):
  """Wraps a TF API Class's class method  in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF Class's class method on KerasTensors.

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = keras.Input(...)
  out = tf.RaggedTensor.from_row_splits(x, y)
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, cls_ref, method_name, **kwargs):
    self.cls_ref = cls_ref
    self.method_name = method_name
    self.cls_symbol = (
        get_canonical_name_for_symbol(
            self.cls_ref, add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(
            self.cls_ref, api_name='keras', add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      kwargs['name'] = K.unique_object_name(
          'tf.' + self.cls_symbol + '.' + self.method_name,
          zero_based=True,
          avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(ClassMethod, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def call(self, args, kwargs):
    return getattr(self.cls_ref, self.method_name)(*args, **kwargs)

  def get_config(self):
    if not self.cls_symbol:
      raise ValueError('This Keras class method conversion tried to convert '
                       'a method belonging to class %s, a class '
                       'that is not an exposed in the TensorFlow API. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.' %
                       self.cls_symbol)
    config = {'cls_symbol': self.cls_symbol, 'method_name': self.method_name}

    base_config = super(ClassMethod, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config.pop('cls_symbol')
    cls_ref = get_symbol_from_name(symbol_name)
    if not cls_ref:
      raise ValueError('TF symbol `tf.%s` could not be found.' % symbol_name)

    config['cls_ref'] = cls_ref

    return cls(**config)
