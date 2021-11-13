
cmake_minimum_required(VERSION 2.8)
project( optflow )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( cv_optflow optflow.cpp )
target_link_libraries( cv_optflow ${OpenCV_LIBS} )

