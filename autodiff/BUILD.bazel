cc_library(
    name = "common",
    hdrs = glob(["common/*.hpp"]),
    visibility = ["//visibility:public"],
    deps = ["@com_github_eigen_eigen//:eigen"],
)

cc_library(
    name = "reverse",
    hdrs = glob([
        "var.hpp",
        "reverse/**/*.hpp",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "@com_github_eigen_eigen//:eigen",
    ],
)

cc_library(
    name = "forward",
    hdrs = glob([
        "dual.hpp",
        "real.hpp",
        "forward/**/*.hpp",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "@com_github_eigen_eigen//:eigen",
    ],
)
