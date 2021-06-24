# Description:
#  Contains the Keras core layers.
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
        "//third_party/tensorflow/python/keras:__subpackages__",
        "//third_party/tensorflow/python/keras/google_deps:keras_layer_usage",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
        "//third_party/tensorflow_models/official/vision/beta/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//third_party/tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "core",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":activation",
        ":activity_regularization",
        ":class_method",
        ":dense",
        ":dropout",
        ":flatten",
        ":instance_method",
        ":instance_property",
        ":keras_op_dispatcher",
        ":lambda",
        ":masking",
        ":permute",
        ":ragged_patches",
        ":repeat_vector",
        ":reshape",
        ":slicing_ops",
        ":sparse_patches",
        ":spatial_dropout_1d",
        ":spatial_dropout_2d",
        ":spatial_dropout_3d",
        ":tf_op_lambda",
    ],
)

tf_py_test(
    name = "core_test",
    size = "medium",
    srcs = ["core_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

py_library(
    name = "activation",
    srcs = ["activation.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "activity_regularization",
    srcs = ["activity_regularization.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "class_method",
    srcs = ["class_method.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "dense",
    srcs = ["dense.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "dropout",
    srcs = ["dropout.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "flatten",
    srcs = ["flatten.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "instance_method",
    srcs = ["instance_method.py"],
    srcs_version = "PY3",
    deps = [
        ":instance_property",
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "instance_property",
    srcs = ["instance_property.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "keras_op_dispatcher",
    srcs = ["keras_op_dispatcher.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "lambda",
    srcs = ["lambda_layer.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "masking",
    srcs = ["masking.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "permute",
    srcs = ["permute.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "ragged_patches",
    srcs = ["ragged_patches.py"],
    srcs_version = "PY3",
    deps = [
        ":class_method",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "repeat_vector",
    srcs = ["repeat_vector.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "slicing_ops",
    srcs = ["slicing_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "sparse_patches",
    srcs = ["sparse_patches.py"],
    srcs_version = "PY3",
    deps = [
        ":instance_method",
        ":instance_property",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "spatial_dropout_1d",
    srcs = ["spatial_dropout_1d.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "spatial_dropout_2d",
    srcs = ["spatial_dropout_2d.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "spatial_dropout_3d",
    srcs = ["spatial_dropout_3d.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "tf_op_lambda",
    srcs = ["tf_op_lambda.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)
