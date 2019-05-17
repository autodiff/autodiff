include_directories(${CMAKE_SOURCE_DIR})

add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(tests
    forward.test.cpp
    forward-v2.test.cpp
    # forward-v3.test.cpp
    reverse.test.cpp
    $<TARGET_OBJECTS:catch>)
