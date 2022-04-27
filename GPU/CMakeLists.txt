cmake_minimum_required(VERSION 3.21)
project(HashSelection CUDA)

file(GLOB TARGETS "../*")
#list(FILTER TARGETS EXCLUDE REGEX main.cu)

set(CMAKE_CUDA_STANDARD 20)
include_directories(../)

add_executable(HashSelection main.cu ${TARGETS})

set_target_properties(HashSelection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET HashSelection PROPERTY CUDA_ARCHITECTURES OFF)