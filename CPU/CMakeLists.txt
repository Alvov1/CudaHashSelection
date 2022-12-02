cmake_minimum_required(VERSION 3.22)
project(HashSelection)

find_package(OpenSSL REQUIRED)
set(CMAKE_CXX_STANDARD 20)

include_directories(../Common ../Common/ScreenWriting)
file (GLOB Targets main.cpp * ../Common/* ../Common/ScreenWriting/*)

if (CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "Note: including file:")
endif()

add_executable(HashSelection ${Targets} ../Common/ScreenWriting/Timer.h ../Common/ScreenWriting/Console.h ../Common/ScreenWriting/ScreenWriter.cpp ../Common/ScreenWriting/Timer.cpp)
if (OPENSSL_FOUND)
    # Add the include directories for compiling
    target_include_directories(HashSelection PRIVATE ${OPENSSL_INCLUDE_DIR})
    # Add the static lib for linking
    target_link_libraries(HashSelection OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "Found OpenSSL ${OPENSSL_VERSION}")
else()
    message(STATUS "OpenSSL Not Found")
endif()
