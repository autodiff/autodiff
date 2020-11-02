find_package(Catch2 REQUIRED)

add_executable(tests
        forward.test.cpp
        reverse.test.cpp
        tests-main.cpp)

target_link_libraries(tests Catch2::Catch2)
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
