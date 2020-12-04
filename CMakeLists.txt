# The minimum required cmake version
cmake_minimum_required(VERSION 3.0)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use ccache to speed up repeated compilations
include(CCache)

# Name and details of the project
project(autodiff VERSION 0.5.13 LANGUAGES CXX)

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Ensure proper configuration if in a conda environment
include(CondaAware)

# Find Eigen3 if its include path is not given
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# Show message stating path to Eigen3
message(STATUS "Using Eigen3 library found at ${EIGEN3_INCLUDE_DIR}")

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(autodiff)
add_subdirectory(examples)
add_subdirectory(test)

# Install the cmake config files that permit users to use find_package(Reaktoro)
include(autodiffInstallCMakeConfigFiles)
