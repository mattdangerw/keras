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
"""Contains the InstanceMethod helper class."""
from keras.layers.core.instance_property import InstanceProperty


class InstanceMethod(InstanceProperty):
  """Wraps an instance method access (e.g.

  `x.foo(arg)` in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` with additional arguments `args` and
  `kwargs` returns `obj.attr_name(*args, **kwargs)`.

  KerasTensors specialized for specific extension types use it to
  represent dynamic instance method calls on the represented object, e.g.

  x = keras.Input(..., ragged=True)
  new_values = keras.Input(...)
  out = x.with_values(new_values)
  """

  def call(self, obj, args, kwargs):
    method = getattr(obj, self.attr_name)
    return method(*args, **kwargs)
