# Collect all source files from the current directory for the Python module
file(GLOB_RECURSE CXX_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cxx)

# Compile autodiff cxx files into a Python module
pybind11_add_module(autodiff4py ${CXX_FILES})

# Add the root directory of the project to the include list
target_include_directories(autodiff4py PRIVATE ${PROJECT_SOURCE_DIR})

# Link autodiff against autodiff C++ library
target_link_libraries(autodiff4py LINK_PRIVATE autodiff::autodiff Eigen3::Eigen)

# This is needed to avoid compilation issues when using clang-9 and c++17
if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 10.0.0)
    target_compile_options(autodiff4py PRIVATE -fsized-deallocation)
endif()
