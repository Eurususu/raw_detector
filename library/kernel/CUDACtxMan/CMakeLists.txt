project(CUDACtxMan)

aux_source_directory(. DIR_SRCS)

cuda_add_library(CUDACtxMan ${DIR_SRCS})

target_include_directories(CUDACtxMan  PUBLIC ../../include)