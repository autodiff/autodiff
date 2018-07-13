include_directories(${CMAKE_SOURCE_DIR})

add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(autodifftest test.cpp $<TARGET_OBJECTS:catch>)

