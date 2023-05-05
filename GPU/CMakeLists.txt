cmake_minimum_required(VERSION 3.22)
#project(HashSelection CUDA)
project(HashSelection)

set(CMAKE_CXX_STANDARD 20)

file(GLOB Sources *.cpp)
add_executable(HashSelection ${Sources})

add_subdirectory(../Hash Hash)
include_directories(../Hash ../Common)

set_target_properties(HashSelection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET HashSelection PROPERTY CUDA_ARCHITECTURES OFF)