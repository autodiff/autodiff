find_package(Catch2 REQUIRED)

add_executable(tests
    main.cpp
    reverse/reverse.test.cpp
    forward/forward.test.cpp
    forward/real/real.test.cpp
    forward/real/eigen.test.cpp)

target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

target_link_libraries(tests Catch2::Catch2)
