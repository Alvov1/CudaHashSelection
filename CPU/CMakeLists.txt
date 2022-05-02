cmake_minimum_required(VERSION 3.21)
project(HashSelection CUDA)

set(CMAKE_CUDA_STANDARD 20)
include_directories(../)

add_executable(HashSelection main.cu
        ../HostHash.h
        ../HostHash.cu
        ../Timer.h
        ../Dictionary.h)

set_target_properties(HashSelection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
