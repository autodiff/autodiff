find_package(Catch2 REQUIRED)

add_executable(tests
    reverse.test.cpp
    forward.test.cpp
    real/real.test.cpp
    real/eigen.test.cpp)

target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

target_link_libraries(tests Catch2::Catch2)
