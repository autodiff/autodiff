include_directories(${CMAKE_SOURCE_DIR})

add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(reverse reverse.cpp $<TARGET_OBJECTS:catch>)
add_executable(forward forward.cpp $<TARGET_OBJECTS:catch>)

