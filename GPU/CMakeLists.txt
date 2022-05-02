cmake_minimum_required(VERSION 3.21)
project(HashSelection CUDA)

set(CMAKE_CUDA_STANDARD 20)
include_directories(../)

add_executable(HashSelection main.cu
        StorageGPU.h
        DeviceHash.h
        DeviceHash.cu
        Array.h
        ../Dictionary.h
        ../Timer.h
        ../HostHash.h
        ../HostHash.cu)

set_target_properties(HashSelection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET HashSelection PROPERTY CUDA_ARCHITECTURES OFF)