cmake_minimum_required(VERSION 3.8)
project(CUDATest LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE PATH "CUDA host compiler" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE PATH "CUDA compiler" FORCE)

include_directories(/usr/local/cuda/include)


include_directories(/home/luca/cuda-samples/Common)

add_executable(Main main.cu)