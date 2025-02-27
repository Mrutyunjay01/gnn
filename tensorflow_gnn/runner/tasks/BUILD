load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "pytype_strict_library")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "py_strict_test")
load("@tensorflow_gnn//tensorflow_gnn:tensorflow_gnn.bzl", "distribute_py_test")

licenses(["notice"])

pytype_strict_library(
    name = "classification",
    srcs = ["classification.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_strict_test(
    name = "classification_test",
    srcs = ["classification_test.py"],
    srcs_version = "PY3",
    deps = [
        ":classification",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "dgi",
    srcs = ["dgi.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

distribute_py_test(
    name = "dgi_test",
    srcs = ["dgi_test.py"],
    srcs_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/238827505)
        "nomultivm",  # TODO(b/170502145)
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":dgi",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "regression",
    srcs = ["regression.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_gnn/runner:__pkg__"],
    deps = [
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
        "//tensorflow_gnn/runner:interfaces",
    ],
)

py_strict_test(
    name = "regression_test",
    srcs = ["regression_test.py"],
    srcs_version = "PY3",
    deps = [
        ":regression",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//tensorflow_gnn",
    ],
)
