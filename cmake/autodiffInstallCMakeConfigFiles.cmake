# The path where cmake config files are installed
set(AUTODIFF_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/autodiff)

install(EXPORT autodiffTargets
    FILE autodiffTargets.cmake
    NAMESPACE autodiff::
    DESTINATION ${AUTODIFF_INSTALL_CONFIGDIR}
    COMPONENT cmake)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/autodiffConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffConfig.cmake
    INSTALL_DESTINATION ${AUTODIFF_INSTALL_CONFIGDIR}
    PATH_VARS AUTODIFF_INSTALL_CONFIGDIR)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffConfigVersion.cmake
    DESTINATION ${AUTODIFF_INSTALL_CONFIGDIR})
