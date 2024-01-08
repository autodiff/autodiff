find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_CPP_FILES "*.test.cpp")
add_executable(autodiff-cpptests main.cpp ${TEST_CPP_FILES})
target_link_libraries(autodiff-cpptests autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
set_target_properties(autodiff-cpptests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE TEST_CUDA_FILES "*.test.cu")
    add_executable(autodiff-cudatests main.cpp ${TEST_CUDA_FILES})
    target_link_libraries(autodiff-cudatests autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
    set_target_properties(autodiff-cudatests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
endif()
# Note: Disabled compilation of implicit conversion tests because they are
# stressing too much CI compilers in addition to the other tests. 
# Allan Leal, 19.06.2023

# add_executable(autodiff-cpptests-implicit-conversion main.cpp ${TEST_CPP_FILES})
# target_compile_definitions(autodiff-cpptests-implicit-conversion PUBLIC AUTODIFF_ENABLE_IMPLICIT_CONVERSION=1)
# target_link_libraries(autodiff-cpptests-implicit-conversion autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
# set_target_properties(autodiff-cpptests-implicit-conversion PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# Add target tests that performs all C++ and Python tests
if(CMAKE_CUDA_COMPILER)
    add_custom_target(tests
        COMMENT "Running C++ and CUDA tests..."
        COMMAND $<TARGET_FILE:autodiff-cpptests>
        COMMAND $<TARGET_FILE:autodiff-cudatests>
        # COMMAND $<TARGET_FILE:autodiff-cpptests-implicit-conversion>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    add_custom_target(tests
            COMMENT "Running C++ tests..."
            COMMAND $<TARGET_FILE:autodiff-cpptests>
            # COMMAND $<TARGET_FILE:autodiff-cpptests-implicit-conversion>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

if(AUTODIFF_TEST_SANITIZE)
    include(CheckCXXCompilerFlag)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")  # Also needs to be a link flag for test to work
    check_cxx_compiler_flag(-fsanitize=address HAVE_ASAN)
    unset(CMAKE_REQUIRED_FLAGS)
    if(HAVE_ASAN)
        target_compile_options(autodiff-cpptests PRIVATE "-fsanitize=address")
        # target_compile_options(autodiff-cpptests-implicit-conversion PRIVATE "-fsanitize=address")
        target_link_options(autodiff-cpptests PRIVATE "-fsanitize=address")
        # target_link_options(autodiff-cpptests-implicit-conversion PRIVATE "-fsanitize=address")
        # if(CMAKE_CUDA_COMPILER)
        #     target_compile_options(autodiff-cudatests PRIVATE "-fsanitize=address")
        #     target_link_options(autodiff-cudatests PRIVATE "-fsanitize=address")
        # endif()
    endif()
endif()
