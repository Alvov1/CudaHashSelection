cmake_minimum_required(VERSION 3.21)
project(HashSelection)

set(CMAKE_CXX_STANDARD 20)

include_directories(../Common)
file (GLOB Targets main.cpp ../Common/*)

add_executable(HashSelection ${Targets})
