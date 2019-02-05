# The minimum required cmake version
cmake_minimum_required(VERSION 3.0)

# Name and details of the project
project(autodiff VERSION 0.3)

# Set the cmake module path of the project
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(autodiff)
add_subdirectory(examples)
add_subdirectory(test)

# Install the cmake config files that permit users to use find_package(Reaktoro)
include(autodiffInstallCMakeConfigFiles)
