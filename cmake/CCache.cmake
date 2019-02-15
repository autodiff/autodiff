# Speed up repeated and similar compilation operations with this cmake module.
#
# This module tries first to find if ccache is installed in the system.
# If found, the cmake variables CMAKE_C_COMPILER_LAUNCHER and
# CMAKE_CXX_COMPILER_LAUNCHER are then set to the ccache program.
# As a result, ccache will be used, together with the chosen
# (or default) compiler to cache the compilations. Subsequent
# compilations can then be performed much faster (e.g., changing
# branches).
#
# Add the option -DCMAKE_CCACHE_IGNORE=TRUE to ignore ccache speed up.
#
# *** IMPORTANT ***
# To use this module, add:
#
#     include(CCache)
#
# before the project command. For example:
#
#     cmake_minimum_required(VERSION 3.0)
#     include(CCache)
#     project(HelloWorld)

# Skip if this file has already been included
if(CCACHE_INCLUDED)
    return()
else()
    set(CCACHE_INCLUDED TRUE)
endif()

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM AND NOT CMAKE_CCACHE_IGNORE)
    message(STATUS "CCache: using ccache to potentially speed up future build tasks.")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()
