cc_library(
    name = "common",
    hdrs = glob(["common/*.hpp"]),
    deps = ["@com_github_eigen_eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "reverse",
    hdrs = glob(["reverse.hpp", "reverse/**/*.hpp"]),
    deps = ["@com_github_eigen_eigen//:eigen",
            ":common"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "forward",
    hdrs = glob(["forward.hpp", "forward/**/*.hpp"]),
    deps = [
        "@com_github_eigen_eigen//:eigen",
        ":common",
    ],
    visibility = ["//visibility:public"],
)
