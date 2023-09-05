cmake_minimum_required(VERSION 3.10)
project(sports-camera-algo)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_compile_options(-fno-use-cxa-atexit -fexceptions -Wno-unused-function -Wno-unused-variable
                    -Wno-sign-compare -Wno-reorder -Wno-unused-but-set-variable -Wno-misleading-indentation
                    -fPIC -fpermissive)
link_directories(/usr/local/cuda/lib64)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)                    
                    
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_86 --default-stream=per-thread)
include_directories(include src ${CUDA_INCLUDE_DIRS} library/include/inline)

add_subdirectory(library/kernel/AQCodec)
add_subdirectory(library/kernel/CUDACtxMan)

cuda_add_executable(test_decoder test/test_decoder.cpp test/Channel.cu)
target_include_directories(test_decoder PUBLIC library/include ${OpenCV_INCLUDE_DIRS} utils)
target_link_libraries(test_decoder ${OpenCV_LIBS} CUDACtxMan AQCodec cuda nvcuvid nvidia-encode rt nppicc)

