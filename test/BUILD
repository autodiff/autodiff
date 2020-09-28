cc_test(
    name = "forward",
    srcs = ["catch.hpp", "forward.test.cpp"],
    deps = [
        "//:autodiff",
    ],
    copts = ["-I./"],
    defines = ["CATCH_CONFIG_MAIN"]
)

cc_test(
    name = "reverse",
    srcs = ["catch.hpp", "reverse.test.cpp"],
    deps = [
        "//:autodiff",
    ],
    copts = ["-I./"]
)