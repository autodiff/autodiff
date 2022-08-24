find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_FILES "*.test.cpp")
add_executable(autodiff-cpptests main.cpp ${TEST_FILES})
target_link_libraries(autodiff-cpptests autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
set_target_properties(autodiff-cpptests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# Add target tests that performs all C++ and Python tests
add_custom_target(tests
    COMMENT "Running C++ tests..."
    COMMAND $<TARGET_FILE:autodiff-cpptests>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

if(AUTODIFF_TEST_SANITIZE)
    include(CheckCXXCompilerFlag)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")  # Also needs to be a link flag for test to work
    check_cxx_compiler_flag(-fsanitize=address HAVE_ASAN)
    unset(CMAKE_REQUIRED_FLAGS)
    if(HAVE_ASAN)
        target_compile_options(autodiff-cpptests PRIVATE "-fsanitize=address")
        target_link_options(autodiff-cpptests PRIVATE "-fsanitize=address")
    endif()
endif()
