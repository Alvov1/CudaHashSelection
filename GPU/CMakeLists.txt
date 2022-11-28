cmake_minimum_required(VERSION 3.21)
project(HashSelection CUDA)

find_package(OpenSSL REQUIRED)
set(CMAKE_CUDA_STANDARD 20)

include_directories(../Common)
file (GLOB Targets main.cpp * ../Common/*)

add_executable(HashSelection ${Targets})

if (OPENSSL_FOUND)
    # Add the include directories for compiling
    target_include_directories(HashSelection PRIVATE ${OPENSSL_INCLUDE_DIR})
    # Add the static lib for linking
    target_link_libraries(HashSelection OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "Found OpenSSL ${OPENSSL_VERSION}")
else()
    message(STATUS "OpenSSL Not Found")
endif()

set_target_properties(HashSelection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET HashSelection PROPERTY CUDA_ARCHITECTURES OFF)