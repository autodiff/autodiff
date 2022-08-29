cc_library(
    name = "common",
    hdrs = glob(["common/*.hpp"]),
    deps = ["@com_github_eigen_eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "reverse",
    hdrs = glob(["var.hpp", "reverse/**/*.hpp"]),
    deps = ["@com_github_eigen_eigen//:eigen",
            ":common"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "forward",
    hdrs = glob(["dual.hpp", "real.hpp", "forward/**/*.hpp"]),
    deps = [
        "@com_github_eigen_eigen//:eigen",
        ":common",
    ],
    visibility = ["//visibility:public"],
)
