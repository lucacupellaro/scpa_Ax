cmake_minimum_required(VERSION 3.10)
project(matlib LANGUAGES C)  # Set language to C



file(GLOB Sources *.h *.c)
add_library(matlib STATIC ${Sources})
target_include_directories(matlib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    )