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
"""Contains the KerasOpDispatcher helper class."""
from keras.engine import keras_tensor
from keras.layers.core.tf_op_lambda import TFOpLambda
import tensorflow.compat.v2 as tf


class KerasOpDispatcher(tf.__internal__.dispatch.GlobalOpDispatcher):
  """A global dispatcher that allows building a functional model with TF Ops."""

  def handle(self, op, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in tf.nest.flatten([args, kwargs])):
      return TFOpLambda(op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED


KerasOpDispatcher().register()
