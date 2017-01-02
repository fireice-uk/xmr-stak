project(xmr-stak-cpu)

cmake_minimum_required(VERSION 2.8.10)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "GCC version must be at least 5.1!")
    endif()
endif()

#SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_C_FLAGS "-DNDEBUG -march=westmere -O3 -m64")
SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "-s")
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "-s")
set(EXECUTABLE_OUTPUT_PATH "bin")

file(GLOB SOURCES "crypto/*.c" "*.cpp")

add_executable(xmr-stak-cpu ${SOURCES})
target_link_libraries(xmr-stak-cpu pthread)
 

