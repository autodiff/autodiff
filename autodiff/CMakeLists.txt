
# Create the autodiff interface library
add_library(autodiff INTERFACE)

# Set autodiff compilation features to be propagated to client code.
target_compile_features(autodiff INTERFACE cxx_std_17)

# target_include_directories(autodiff INTERFACE ${CMAKE_INSTALL_INCLUDEDIR})
# Add the include paths to the Reaktoro target
target_include_directories(autodiff
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install autodiff interface library
install(TARGETS autodiff EXPORT autodiffTargets)

# Install autodiff header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/autodiff
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers
    PATTERN "CMakeLists.txt" EXCLUDE)
