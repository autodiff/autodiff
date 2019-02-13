include_directories(${CMAKE_SOURCE_DIR})

add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(tests
    forward.test.cpp
    reverse.test.cpp
    $<TARGET_OBJECTS:catch>)
