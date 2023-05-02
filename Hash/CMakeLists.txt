cmake_minimum_required(VERSION 3.13)
project(MutationsTest)
set(CMAKE_CXX_STANDARD 20)

file(GLOB Sources *.cpp)

add_executable(MutationsTest ${Sources})

find_package(OpenSSL REQUIRED)
if (OPENSSL_FOUND)
    # Add the include directories for compiling
    target_include_directories(MutationsTest PRIVATE ${OPENSSL_INCLUDE_DIR})
    # Add the static lib for linking
    target_link_libraries(MutationsTest OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "Found OpenSSL ${OPENSSL_VERSION}")
else()
    message(STATUS "OpenSSL Not Found")
endif()