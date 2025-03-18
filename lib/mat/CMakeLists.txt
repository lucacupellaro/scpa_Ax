cmake_minimum_required(VERSION 3.10)
project(matLib VERSION 0.1)
file(GLOB Sources *.h *.c)
add_library(matLib STATIC ${Sources})
target_include_directories(matLib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    )