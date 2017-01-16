project(xmr-stak-cpu)

cmake_minimum_required(VERSION 2.8.10)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "GCC version must be at least 5.1!")
    endif()
endif()

find_library(MHTD NAMES microhttpd)
if("${MHTD}" STREQUAL "MHTD-NOTFOUND")
    message(FATAL_ERROR "libmicrohttpd is required")
endif()

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CONFIGURATION_TYPES "RELEASE;STATIC")
if("${CMAKE_BUILD_TYPE}" STREQUAL "") 
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_C_FLAGS "-DNDEBUG -march=westmere -O3 -m64 -s")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11")

set(CMAKE_EXE_LINKER_FLAGS_RELSEASE "")
set(CMAKE_EXE_LINKER_FLAGS_STATIC "-static-libgcc -static-libstdc++")

set(EXECUTABLE_OUTPUT_PATH "bin")

file(GLOB SOURCES "crypto/*.c" "*.cpp")

add_executable(xmr-stak-cpu ${SOURCES})
target_link_libraries(xmr-stak-cpu pthread microhttpd)
 

