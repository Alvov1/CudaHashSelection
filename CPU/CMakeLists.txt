cmake_minimum_required(VERSION 3.22)
#project(HashSelection CUDA)
project(HashSelection)

set(CMAKE_CXX_STANDARD 20)

file(GLOB Sources *.cpp)
add_executable(HashSelection ${Sources})

add_subdirectory(../Hash Hash)
include_directories(../Hash)

find_package(OpenSSL REQUIRED)
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