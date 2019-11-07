add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(tests
    forward.test.cpp
    reverse.test.cpp
    $<TARGET_OBJECTS:catch>)
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
