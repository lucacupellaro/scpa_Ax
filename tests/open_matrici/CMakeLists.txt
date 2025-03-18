cmake_minimum_required(VERSION 3.25)
project(openM VERSION 0.1)

add_subdirectory(../../lib/mat mat_lib_build) 

#target_include_directories(testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/test)
add_executable(main apriMatrice.c)
target_link_libraries(main  PRIVATE matLib)
target_include_directories(main PRIVATE ../../lib/mat)  # Include library headers