
cmake_minimum_required(VERSION 2.8)
project( erosion )

find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
cuda_add_executable(test_er erosion_test.cpp)
target_link_libraries( test_er ${OpenCV_LIBS} -lnppim)

