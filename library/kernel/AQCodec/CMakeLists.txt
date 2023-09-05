project(AQCodec)
include_directories(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(../../include)

file(GLOB_RECURSE SRC_Codec *.cpp *.cu )
cuda_add_library(AQCodec ${SRC_Codec})

