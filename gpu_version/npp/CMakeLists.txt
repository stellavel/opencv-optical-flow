
cmake_minimum_required(VERSION 2.8)
project( optflow )

find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( npp_optflow npp_optflow.cpp )
target_link_libraries( npp_optflow ${OpenCV_LIBS} -lnppim -lnppial -lnppif -lnppitc -lcudart)

