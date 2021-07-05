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
    message(STATUS "CondaAware: Conda environment detected!")
    message(STATUS "CondaAware: Found environment variable CONDA_PREFIX=$ENV{CONDA_PREFIX}")

    # Check if environment variable PYTHON is defined, and if so, set PYTHON_EXECUTABLE to PYTHON
    if(DEFINED ENV{PYTHON})
        message(STATUS "CondaAware: Found environment variable PYTHON=$ENV{PYTHON}")
        message(STATUS "CondaAware: Setting PYTHON_EXECUTABLE to PYTHON=$ENV{PYTHON}")
        set(PYTHON_EXECUTABLE $ENV{PYTHON})
    endif()

    # Ensure python executable is properly selected if not specified yet using
    # cmake variables PYTHON_EXECUTABLE or environment variable PYTHON. The
    # code below tries to set PYTHON_EXECUTABLE with the python executable in
    # the activated conda environment and not in the base environment given by
    # the environment variable CONDA_PYTHON_EXE!
    if(NOT DEFINED PYTHON_EXECUTABLE)
        if(UNIX)
            message(STATUS "CondaAware: Setting PYTHON_EXECUTABLE=$ENV{CONDA_PREFIX}/bin/python")
            set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
        endif()

        if(WIN32)
            message(STATUS "CondaAware: Setting PYTHON_EXECUTABLE=$ENV{CONDA_PREFIX}\\python.exe")
            set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}\\python.exe")
        endif()

        if(NOT DEFINED PYTHON_EXECUTABLE)
            message(FATAL_ERROR "CondaAware: Could not determine a value for PYTHON_EXECUTABLE. Expecting Unix or Windows systems.")
        endif()
    endif()

    # Set auxiliary variable CONDA_AWARE_PREFIX as according to the logic below.
    if(DEFINED ENV{CONDA_BUILD})
        message(STATUS "CondaAware: Detected conda build task (e.g., in a conda-forge build)!")
        if(UNIX)
            set(CONDA_AWARE_PREFIX "$ENV{PREFIX}")
        endif()
        if(WIN32)
            set(CONDA_AWARE_PREFIX "$ENV{LIBRARY_PREFIX}")
        endif()
    else()
        if(UNIX)
            set(CONDA_AWARE_PREFIX "$ENV{CONDA_PREFIX}")
        endif()

        if(WIN32)
            set(CONDA_AWARE_PREFIX "$ENV{CONDA_PREFIX}\\Library")
        endif()
    endif()

    # Check if CONDA_AWARE_PREFIX has been successfully set
    if(DEFINED CONDA_AWARE_PREFIX)
        message(STATUS "CondaAware: Set CONDA_AWARE_PREFIX=${CONDA_AWARE_PREFIX}")
    else()
        message(FATAL_ERROR "CondaAware: Could not determine a value for CONDA_AWARE_PREFIX. Expecting Unix or Windows systems.")
    endif()

    # Set CMAKE_INSTALL_PREFIX to CONDA_AWARE_PREFIX if not specified by the user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "CondaAware: Setting CMAKE_INSTALL_PREFIX=CONDA_AWARE_PREFIX=${CONDA_AWARE_PREFIX}")
        set(CMAKE_INSTALL_PREFIX ${CONDA_AWARE_PREFIX})
    endif()

    # Ensure dependencies from the conda environment are used instead of those from the system.
    list(APPEND CMAKE_PREFIX_PATH ${CONDA_AWARE_PREFIX})
    message(STATUS "CondaAware: Appended ${CONDA_AWARE_PREFIX} to CMAKE_PREFIX_PATH")

    # Ensure include directory in conda environment is known to the project
    include_directories(${CONDA_AWARE_PREFIX}/include)
    message(STATUS "CondaAware: Appended ${CONDA_AWARE_PREFIX}/include to include directories")

    # Ensure library directory in conda environment is known to the project
    link_directories(${CONDA_AWARE_PREFIX}/lib)
    message(STATUS "CondaAware: Appended ${CONDA_AWARE_PREFIX}/lib to link directories")
endif()
