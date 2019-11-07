find_package(Catch2 REQUIRED)

file(GLOB_RECURSE TEST_FILES "*.test.cpp")

add_executable(tests main.cpp ${TEST_FILES})

target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

target_compile_features(tests PRIVATE cxx_std_17)

target_link_libraries(tests Catch2::Catch2)
