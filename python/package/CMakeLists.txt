# The name of the compiled python module file as defined by pybind11 used in setup.py.in
set(AUTODIFF_PYTHON_MODULE_FILENAME autodiff4py${PYTHON_MODULE_EXTENSION})

# The path in the binary dir where the python package is assembled
set(AUTODIFF_PYTHON_PACKAGE_PATH ${CMAKE_CURRENT_BINARY_DIR}/autodiff)

# Configure the setup.py file
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY ${AUTODIFF_PYTHON_PACKAGE_PATH}/setup.py) # ensure setup.py is inside the package dir
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# Create a custom target to build the python package during build stage
add_custom_target(autodiff-setuptools ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/autodiff ${AUTODIFF_PYTHON_PACKAGE_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:autodiff4py> ${AUTODIFF_PYTHON_PACKAGE_PATH}
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} --quiet install --force --prefix=${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${AUTODIFF_PYTHON_PACKAGE_PATH})

# Set dependencies of the python package target
add_dependencies(autodiff-setuptools autodiff4py)

# Set the path where the python package is installed to CMAKE_INSTALL_PREFIX if not given
if(NOT DEFINED AUTODIFF_PYTHON_INSTALL_PREFIX)
    set(AUTODIFF_PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

# If the path is already in Windows format (with backslashes), it can't be
# added directly to the string below. CMake complains about "Invalid escape sequence".
file(TO_CMAKE_PATH "${AUTODIFF_PYTHON_INSTALL_PREFIX}" AUTODIFF_PYTHON_INSTALL_PREFIX)

# Install the python package using setuptools
install(CODE
"
    file(TO_NATIVE_PATH \"${AUTODIFF_PYTHON_INSTALL_PREFIX}\" AUTODIFF_PYTHON_INSTALL_PREFIX_NATIVE)

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=\${AUTODIFF_PYTHON_INSTALL_PREFIX_NATIVE}
        WORKING_DIRECTORY ${AUTODIFF_PYTHON_PACKAGE_PATH})
")
