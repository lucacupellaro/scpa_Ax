cmake_minimum_required(VERSION 3.10)
project(test VERSION 0.1)
file(GLOB Sources *.h *.c)
add_library(test STATIC ${Sources})
target_include_directories(test
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    )