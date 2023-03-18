cmake_minimum_required(VERSION 3.13)
project(simple_example)
set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES main.cpp)

add_executable(simple_example ${SOURCE_FILES})