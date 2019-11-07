find_package(Catch2 REQUIRED)

file(GLOB_RECURSE TEST_FILES "*.test.cpp")

add_executable(tests main.cpp ${TEST_FILES})

target_include_directories(tests PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(tests autodiff Catch2::Catch2)

set_target_properties(tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
