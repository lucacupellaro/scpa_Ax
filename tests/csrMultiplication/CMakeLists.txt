cmake_minimum_required(VERSION 3.25)
project(openM VERSION 0.1)

find_package(OpenMP REQUIRED)

# Set OpenMP flags globally for all targets
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp")
set(CMAKE_BUILD_TYPE Debug)

# Add and configure the `matLib` library
add_subdirectory(../../lib/mat mat_lib_build)

# Compile `matLib` with OpenMP
if(OpenMP_C_FOUND)
    target_link_libraries(matlib PUBLIC OpenMP::OpenMP_C)
    target_compile_options(matlib PRIVATE -fopenmp)
endif()

# Create the `Main` executable
add_executable(Main main.c)

# Compile `Main` with OpenMP
if(OpenMP_C_FOUND)
    target_compile_options(Main PRIVATE -fopenmp)
    target_link_libraries(Main PUBLIC OpenMP::OpenMP_C)
endif()

# Link `matLib` to `Main`
target_link_libraries(Main PRIVATE matlib)

# Include `matLib` headers
target_include_directories(Main PRIVATE ../../lib/mat)
