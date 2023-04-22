cmake_minimum_required(VERSION 3.13)
project(MutationsTest)
set(CMAKE_CXX_STANDARD 20)

file(GLOB Sources *.cpp)

add_executable(MutationsTest ${Sources})