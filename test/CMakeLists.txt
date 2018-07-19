include_directories(${CMAKE_SOURCE_DIR}/autodiff)

add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(testsuite test.cpp $<TARGET_OBJECTS:catch>)

