cmake_minimum_required(VERSION 3.22)
project(HashSelection CUDA)

set(CMAKE_CXX_STANDARD 20)

include_directories(../Hash ../Common)

add_executable(HashSelection main.cu)

add_subdirectory(../Hash Hash)
add_subdirectory(../Common Common)

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