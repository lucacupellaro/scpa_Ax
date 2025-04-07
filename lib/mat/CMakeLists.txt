cmake_minimum_required(VERSION 3.25)
project(matlib LANGUAGES C)  # Set language to C
enable_language(CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



file(GLOB Sources *.h *.c)
add_library(matlib STATIC ${Sources})
target_include_directories(matlib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    )