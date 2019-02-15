# This cmake module determines if a conda environment is active.
#
# Note: CMAKE_INSTALL_PREFIX is set to the environment variable CONDA_PREFIX
# if a conda environment is found active.
#
# This automatic behavior can be overriden by manually
# specifying a different CMAKE_INSTALL_PREFIX.

# Skip if this file has already been included
if(CONDA_AWARE_INCLUDED)
    return()
else()
    set(CONDA_AWARE_INCLUDED TRUE)
endif()

# Check if a conda environment is active
if(DEFINED ENV{CONDA_PREFIX})
    # Show that conda env has been recognized
    message(STATUS "CondaAware: conda environment recognized!")
    message(STATUS "CondaAware: CONDA_PREFIX=$ENV{CONDA_PREFIX}")

    # Check if in Unix and not in a conda build task
    if(UNIX AND NOT DEFINED ENV{CONDA_BUILD})
        set(CONDA_AWARE_PREFIX "$ENV{CONDA_PREFIX}")
        message(STATUS "CondaAware: CONDA_AWARE_PREFIX=CONDA_PREFIX (${CONDA_AWARE_PREFIX})")
    endif()

    # Check if in Unix and not in a conda build task
    if(WIN32 AND NOT DEFINED ENV{CONDA_BUILD})
        set(CONDA_AWARE_PREFIX "$ENV{CONDA_PREFIX}\\Library")
        message(STATUS "CondaAware: CONDA_AWARE_PREFIX=CONDA_PREFIX\\Library (${CONDA_AWARE_PREFIX})")
    endif()

    # Check if in Unix and in a conda build task
    if(UNIX AND DEFINED ENV{CONDA_BUILD})
        set(CONDA_AWARE_PREFIX "$ENV{PREFIX}")
        message(STATUS "CondaAware: conda build task recognized!")
        message(STATUS "CondaAware: CONDA_AWARE_PREFIX=PREFIX (${CONDA_AWARE_PREFIX})")
    endif()

    # Check if in Windows and in a conda build task
    if(WIN32 AND DEFINED ENV{CONDA_BUILD})
        set(CONDA_AWARE_PREFIX "$ENV{LIBRARY_PREFIX}")
        message(STATUS "CondaAware: conda build task recognized!")
        message(STATUS "CondaAware: CONDA_AWARE_PREFIX=LIBRARY_PREFIX (${CONDA_AWARE_PREFIX})")
    endif()

    # Fatal error if none of the previous checks succeeded
    if(NOT DEFINED CONDA_AWARE_PREFIX)
        message(FATAL_ERROR "Could not determine a value for CONDA_AWARE_PREFIX")
    endif()

    # Set CMAKE_INSTALL_PREFIX to CONDA_AWARE_PREFIX if not specified by the user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "CondaAware: CMAKE_INSTALL_PREFIX set to CONDA_AWARE_PREFIX")
        set(CMAKE_INSTALL_PREFIX ${CONDA_AWARE_PREFIX})
    endif()

    # Ensure dependencies from the conda environment are used instead of those from the system.
    message(STATUS "CondaAware: appended CONDA_AWARE_PREFIX to CMAKE_PREFIX_PATH")
    list(APPEND CMAKE_PREFIX_PATH ${CONDA_AWARE_PREFIX})

    # Ensure include directory in conda environment is known to the project
    message(STATUS "CondaAware: appended CONDA_AWARE_PREFIX/include to include directories")
    include_directories(${CONDA_AWARE_PREFIX}/include)

    # Ensure library directory in conda environment is known to the project
    message(STATUS "CondaAware: appended CONDA_AWARE_PREFIX/lib to link directories")
    link_directories(${CONDA_AWARE_PREFIX}/lib)
endif()
