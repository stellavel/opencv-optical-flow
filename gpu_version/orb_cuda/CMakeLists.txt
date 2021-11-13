
cmake_minimum_required(VERSION 2.8)
project( optflow )

find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( orb_cuda orb_cuda.cpp )
target_link_libraries( orb_cuda ${OpenCV_LIBS})

