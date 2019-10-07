add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(tests
    reverse.test.cpp
    forward.test.cpp
    real.real.test.cpp
    real.eigen.test.cpp
    $<TARGET_OBJECTS:catch>)

target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
