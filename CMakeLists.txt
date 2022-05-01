# The minimum required cmake version
cmake_minimum_required(VERSION 3.10)

# Add cmake modules of this project to the module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use ccache to speed up repeated compilations
include(CCache)

# Name and details of the project
project(autodiff VERSION 0.6.7 LANGUAGES CXX)

# Enable parallel build if MSVC is used
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Ensure proper configuration if in a conda environment
include(CondaAware)

# Define build options
option(AUTODIFF_BUILD_TESTS "Enable the compilation of the test files." ON)
option(AUTODIFF_BUILD_PYTHON "Enable the compilation of the python bindings." ON)
option(AUTODIFF_BUILD_EXAMPLES "Enable the compilation of the example files." ON)
option(AUTODIFF_BUILD_DOCS "Enable the build of the documentation and website." ON)

# External libraries
include(FetchContent)
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.3
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)          # ensure cmake instructs compiler to use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # ensure the C++ standard given before is actually used
set(CMAKE_CXX_EXTENSIONS OFF)       # avoid compile flags of the type -std=gnu++1z added by cmake

# Build the library (actually, just provide details about the library target, specify required compile options, etc. because autodiff is header-only)
FetchContent_MakeAvailable(eigen)
if (NOT TARGET eigen)
    message( FATAL_ERROR "eigen requirement could not be found.")
endif()
add_subdirectory(autodiff)

# Build the tests
if(AUTODIFF_BUILD_TESTS)
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v2.13.8
    )
    FetchContent_MakeAvailable(Catch2)
    add_subdirectory(tests)
endif()

# Build the python wrappers
if(AUTODIFF_BUILD_PYTHON)
    set(PYBIND11_CPP_STANDARD -std=c++17)  # Ensure pybind11 uses C++17 standard
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.7.1
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    add_subdirectory(python)
endif()

# Build the examples
if(AUTODIFF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build the docs
if(AUTODIFF_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Install the cmake config files that permit users to use find_package(autodiff)
include(autodiffInstallCMakeConfigFiles)
