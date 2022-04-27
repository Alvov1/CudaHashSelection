cmake_minimum_required(VERSION 3.21)
project(CudaHashSelection CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaHashSelection main.cu)

set_target_properties(CudaHashSelection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
