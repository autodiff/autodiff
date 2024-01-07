# Create the autodiff interface library
add_library(autodiff INTERFACE)

# Add an alias autodiff::autodiff to the target library autodiff
add_library(autodiff::autodiff ALIAS autodiff)

# Set autodiff compilation features to be propagated to client code.
target_compile_features(autodiff INTERFACE cxx_std_17)

# Add the include paths to the Reaktoro target
target_include_directories(autodiff
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CUDA_COMPILER)
    target_compile_options(autodiff INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
endif()

# Install autodiff interface library
install(TARGETS autodiff EXPORT autodiffTargets)

# Install autodiff header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/autodiff
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers
    PATTERN "CMakeLists.txt" EXCLUDE)
