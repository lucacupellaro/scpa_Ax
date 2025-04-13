cmake_minimum_required(VERSION 3.25)
project(matlib LANGUAGES C CXX CUDA)
enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


file(GLOB Sources *.h *.c *.cu)

add_library(matlib STATIC ${Sources})
set_target_properties(matlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matlib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(matlib
    PUBLIC cudart
)
