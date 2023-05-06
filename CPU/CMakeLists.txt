cmake_minimum_required(VERSION 3.22)
project(HashSelection CUDA)

set(CMAKE_CXX_STANDARD 20)

include_directories(../Common)
include_directories(../Hash)

file(GLOB Sources *.cpp *.cu)
add_executable(HashSelection ${Sources})

add_subdirectory(../Common Common)
add_subdirectory(../Hash Hash)

find_package(OpenSSL REQUIRED)
if (OPENSSL_FOUND)
    target_include_directories(HashSelection PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_link_libraries(HashSelection OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "Found OpenSSL ${OPENSSL_VERSION}")
else()
    message(STATUS "OpenSSL Not Found")
endif()

set_target_properties(HashSelection PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(HashSelection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET HashSelection PROPERTY CUDA_ARCHITECTURES OFF)