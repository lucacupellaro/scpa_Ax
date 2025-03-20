cmake_minimum_required(VERSION 3.25)
project(openM VERSION 0.1)

set(CMAKE_BUILD_TYPE Debug)
add_subdirectory(../../lib/mat mat_lib_build) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g ")
#target_include_directories(testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/test)
add_executable(Main apriMatrice.c)
target_compile_definitions(Main PRIVATE)
target_link_libraries(Main  PRIVATE  matLib)
target_include_directories( Main PRIVATE  ../../lib/mat)  # Include library headers