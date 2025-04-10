cmake_minimum_required(VERSION 3.25)
project(K9 LANGUAGES CUDA CXX)
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Set the desired C++ standard for CUDA
set(CMAKE_CUDA_STANDARD 17)  # Use C++17 for CUDA code

# Set the CUDA compute architecture for RTX 40 series (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 80)  # sm_80 for RTX 40 series
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add mat_lib_build from subdirectory
add_subdirectory(../../lib/mat matlib_dir)

# Add the CUDA executable
add_executable(main costantMem.cu)
target_compile_options(main PRIVATE "--generate-line-info")
target_include_directories(main PRIVATE ../../lib/mat)

#set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the mat_lib_build library
target_link_libraries(main PRIVATE matlib)
target_include_directories(main PRIVATE ../../lib/mat)
target_link_directories(main PRIVATE ${CMAKE_BINARY_DIR}/matlib_dir)

add_dependencies(main matlib)
