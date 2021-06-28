# The name of the compiled python module file as defined by pybind11 used in setup.py.in
set(AUTODIFF_PYTHON_MODULE_FILENAME autodiff4py${PYTHON_MODULE_EXTENSION})

# The path in the binary dir where the python package is assembled before it is built
set(AUTODIFF_PYTHON_PACKAGE_PATH ${CMAKE_CURRENT_BINARY_DIR}/autodiff)

# Configure the setup.py file
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# This is needed in Windows so that the next step produces the python package in CMAKE_BINARY_DIR\Lib\site-packages\autodiff
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} CMAKE_BINARY_DIR_NATIVE)

# Create a custom target to build the python package during build stage
add_custom_target(autodiff-setuptools ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/autodiff ${CMAKE_CURRENT_BINARY_DIR}/autodiff
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:autodiff4py> ${CMAKE_CURRENT_BINARY_DIR}/autodiff
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} --quiet install --force --no-compile --prefix=installed/$<CONFIG>
    COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_BINARY_DIR}/autodiff/$<TARGET_FILE_NAME:autodiff4py>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Set dependencies of the python package target
add_dependencies(autodiff-setuptools autodiff4py)

# Ensure the path where the python package is installed is not empty
if(NOT DEFINED AUTODIFF_PYTHON_INSTALL_PREFIX)
    set(AUTODIFF_PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

# Create an install target for the autodiff python package
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/installed/$<CONFIG>/
    DESTINATION ${AUTODIFF_PYTHON_INSTALL_PREFIX})
