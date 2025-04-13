cmake_minimum_required(VERSION 3.8)
project(CUDATest LANGUAGES C CXX CUDA)


set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE PATH "CUDA host compiler" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE PATH "CUDA compiler" FORCE)
#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(USE_CUDA_ALLOC)
    add_definitions(-D USE_CUDA_ALLOC)
endif()


include_directories(/usr/local/cuda/include)
include_directories(/home/luca/cuda-samples/Common)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/mat)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/cuda)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})


add_executable(Main 
    main.cu 
    ../../lib/mat/mmio.c
    ../../lib/mat/matriciOpp.c
    ../../lib/mat/random_vector.c 
    ../../lib/mat/mmio.c          
    ../../lib/mat/hll.c           
   
    
)
