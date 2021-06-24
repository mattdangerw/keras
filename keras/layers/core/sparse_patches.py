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
"""Patches the sparse tensor class."""
from keras.engine import keras_tensor
from keras.layers.core.instance_method import InstanceMethod
from keras.layers.core.instance_property import InstanceProperty


def _delegate_property(keras_tensor_cls, property_name):  # pylint: disable=invalid-name
  """Register property on a KerasTensor class.

  Calling this multiple times with the same arguments should be a no-op.

  This method exposes a property on the KerasTensor class that will use an
  `InstanceProperty` layer to access the property on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    property_name: The name of the property to expose and delegate to the
      represented (Composite)Tensor.
  """
  # We use a lambda because we can't create a Keras layer at import time
  # due to dynamic layer class versioning.
  property_access = property(lambda self: InstanceProperty(property_name)(self))  # pylint: disable=unnecessary-lambda
  setattr(keras_tensor_cls, property_name, property_access)


def _delegate_method(keras_tensor_cls, method_name):  # pylint: disable=invalid-name
  """Register method on a KerasTensor class.

  Calling this function times with the same arguments should be a no-op.

  This method exposes an instance method on the KerasTensor class that will use
  an `InstanceMethod` layer to run the desired method on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    method_name: The name of the method to expose and delegate to the
      represented (Composite)Tensor.
  """

  def delegate(self, *args, **kwargs):
    return InstanceMethod(method_name)(self, args, kwargs)

  setattr(keras_tensor_cls, method_name, delegate)


# We do not support the `uniform_row_length` property because it
# returns either `None` or an int tensor, and code that relies on it tends
# to check `is None` directly. Delegating it here would always return a
# `KerasTensor`, regardless of what can be statically inferred. This would
# never equal `None`, breaking code that expects it to be partially-static
# in unpredictable ways.
for ragged_property in [
    'values', 'flat_values', 'row_splits', 'nested_row_splits'
]:
  _delegate_property(keras_tensor.RaggedKerasTensor, ragged_property)

for ragged_method_name in [
    'value_rowids',
    'nested_value_rowids',
    'nrows',
    'row_starts',
    'row_limits',
    'row_lengths',
    'nested_row_lengths',
    'bounding_shape',
    'with_values',
    'with_flat_values',
    'with_row_splits_dtype',
    'merge_dims',
    'to_tensor',
    'to_sparse',
]:
  _delegate_method(keras_tensor.RaggedKerasTensor, ragged_method_name)

for sparse_property in [
    'indices',
    'values',
]:
  _delegate_property(keras_tensor.SparseKerasTensor, sparse_property)

for sparse_method in [
    'with_values',
]:
  _delegate_method(keras_tensor.SparseKerasTensor, sparse_method)
