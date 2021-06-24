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
"""Patches the ragged tensor class."""
from keras.engine import keras_tensor
from keras.layers.core.class_method import ClassMethod
import tensorflow.compat.v2 as tf


class TFClassMethodDispatcher(tf.__internal__.dispatch.OpDispatcher):
  """A class method dispatcher that allows building a functional model with TF class methods."""

  def __init__(self, cls, method_name):
    self.cls = cls
    self.method_name = method_name

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in tf.nest.flatten([args, kwargs])):
      return ClassMethod(self.cls, self.method_name)(args[1:], kwargs)
    else:
      return self.NOT_SUPPORTED


for ragged_class_method in [
    'from_value_rowids',
    'from_row_splits',
    'from_row_lengths',
    'from_row_starts',
    'from_row_limits',
    'from_uniform_row_length',
    'from_nested_value_rowids',
    'from_nested_row_splits',
    'from_nested_row_lengths',
    'from_tensor',
    'from_sparse',
]:
  TFClassMethodDispatcher(tf.RaggedTensor, ragged_class_method).register(
      getattr(tf.RaggedTensor, ragged_class_method))
