cmake_minimum_required(VERSION 3.25)
project(testing VERSION 0.1)

add_subdirectory(../../lib/test test_lib_build) 

#target_include_directories(testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/test)
add_executable(Main main.c)
target_link_libraries(Main  PRIVATE test)
target_include_directories(Main PRIVATE ../../lib/test)  # Include library headers