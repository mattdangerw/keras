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
"""Contains the InstanceProperty helper class."""
from keras import backend as K
from keras.engine.base_layer import Layer
import tensorflow.compat.v2 as tf


class InstanceProperty(Layer):
  """Wraps an instance property access (e.g.

  `x.foo`) in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` returns `obj.attr_name`.

  KerasTensors specialized for specific extension types use it to
  represent instance property accesses on the represented object in the
  case where the property needs to be dynamically accessed as opposed to
  being statically computed from the typespec, e.g.

  x = keras.Input(..., ragged=True)
  out = x.flat_values
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, attr_name, **kwargs):
    self.attr_name = attr_name

    if 'name' not in kwargs:
      kwargs['name'] = K.unique_object_name(
          'input.' + self.attr_name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(InstanceProperty, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

  def call(self, obj):
    return getattr(obj, self.attr_name)

  def get_config(self):
    config = {'attr_name': self.attr_name}
    base_config = super(InstanceProperty, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
