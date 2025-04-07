cmake_minimum_required(VERSION 3.25)  # Ensure CMake supports CUDA
project(MyCudaProject LANGUAGES CUDA CXX)

# Set CUDA as a required language
enable_language(CUDA)
find_package(CUDA REQUIRED)
# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add the CUDA executable
add_executable(main main.cu)

# Set CUDA compilation flags if needed
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# If you need to link additional libraries, add them here
# target_link_libraries(main PRIVATE some_library)
