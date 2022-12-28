# Configure the setup.py file
configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# Create a custom target to build the python package during build stage
add_custom_target(autodiff-setuptools ALL
    DEPENDS autodiff4py
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/autodiff ${CMAKE_CURRENT_BINARY_DIR}/autodiff
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:autodiff4py> ${CMAKE_CURRENT_BINARY_DIR}/autodiff
    COMMAND ${PYTHON_EXECUTABLE} setup.py --quiet build --force
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/autodiff/$<TARGET_FILE_NAME:autodiff4py>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Ensure the path where the python package is installed is not empty
if(NOT DEFINED AUTODIFF_PYTHON_INSTALL_PREFIX)
    file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX} AUTODIFF_PYTHON_INSTALL_PREFIX)
endif()

# Create an install target for the autodiff python package
install(CODE "execute_process(COMMAND pip install . --no-deps --force-reinstall --no-compile --prefix=${AUTODIFF_PYTHON_INSTALL_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
