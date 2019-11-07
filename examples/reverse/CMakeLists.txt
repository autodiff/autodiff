file(GLOB_RECURSE CPPFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(CPPFILE ${CPPFILES})
    get_filename_component(CPPNAME ${CPPFILE} NAME_WE)
    add_executable(${CPPNAME} ${CPPFILE})
    target_compile_features(${CPPNAME} PRIVATE cxx_std_17)
    target_include_directories(${CPPNAME} PRIVATE ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
endforeach()
