# Create the autodifflib interface library
add_library(autodifflib INTERFACE)

# Add an alias autodiff::autodiff to the target library autodifflib
add_library(autodiff::autodiff ALIAS autodifflib)

# Set autodifflib compilation features to be propagated to client code.
target_compile_features(autodifflib INTERFACE cxx_std_17)

# Add the include paths to the Reaktoro target
target_include_directories(autodifflib
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install autodifflib interface library
install(TARGETS autodifflib EXPORT autodiffTargets)

# Install autodifflib header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/autodifflib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers
    PATTERN "CMakeLists.txt" EXCLUDE)
