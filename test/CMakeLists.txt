include_directories(${CMAKE_SOURCE_DIR}/autodiff)

add_library(catch OBJECT catch.hpp catch.cpp)

add_executable(var var.cpp $<TARGET_OBJECTS:catch>)
add_executable(dual dual.cpp $<TARGET_OBJECTS:catch>)

