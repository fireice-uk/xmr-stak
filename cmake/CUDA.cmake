get_filename_component(DEVICE_COMPILER "${CMAKE_CUDA_COMPILER}" NAME)
set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    list(APPEND DEVICE_COMPILER "clang")
endif()

set_property(CACHE CUDA_COMPILER PROPERTY STRINGS "${DEVICE_COMPILER}")

if(CUDA_LARGEGRID)
    list(APPEND CUDA_NVCC_FLAGS "-DCUDA_LARGEGRID=${CUDA_LARGEGRID}")
endif()

if(CUDA_COMPILER STREQUAL "clang")
    set(CLANG_BUILD_FLAGS "-O3 -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    # activation usage of FMA
    set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -ffp-contract=fast")

    if(CUDA_SHOW_REGISTER)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -Xcuda-ptxas -v")
    endif(CUDA_SHOW_REGISTER)

    if(CUDA_KEEP_FILES)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -save-temps=${PROJECT_BINARY_DIR}")
    endif(CUDA_KEEP_FILES)

    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architectures
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} --cuda-gpu-arch=sm_${CUDA_ARCH_ELEM}")
    endforeach()
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR CUDA_COMPILER STREQUAL "nvcc")
    # add c++11 for cuda
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architecture
        if(21 EQUAL CUDA_ARCH_ELEM)
            list(APPEND CUDA_NVCC_FLAGS
                    "-gencode=arch=compute_20,code=sm_${CUDA_ARCH_ELEM}")
        else()
            list(APPEND CUDA_NVCC_FLAGS
                    "-gencode=arch=compute_${CUDA_ARCH_ELEM},code=sm_${CUDA_ARCH_ELEM}")
            list(APPEND CUDA_NVCC_FLAGS
                    "-gencode=arch=compute_${CUDA_ARCH_ELEM},code=compute_${CUDA_ARCH_ELEM}")
        endif()
    endforeach()

    # give each thread an independent default stream
    list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")

    if(CUDA_SHOW_REGISTER)
        list(APPEND CUDA_NVCC_FLAGS "-Xptxas=-v")
    endif(CUDA_SHOW_REGISTER)

    if(CUDA_SHOW_CODELINES)
        list(APPEND CUDA_NVCC_FLAGS "--source-in-ptx" "-lineinfo")
        set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif(CUDA_SHOW_CODELINES)

    if(CUDA_KEEP_FILES)
        list(APPEND CUDA_NVCC_FLAGS "--keep" "--keep-dir ${PROJECT_BINARY_DIR}")
    endif(CUDA_KEEP_FILES)

    if(CUDA_VERSION VERSION_LESS 8.0)
        # avoid that nvcc in CUDA < 8 tries to use libc `memcpy` within the kernel
        list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES")
        # for CUDA 7.5 fix compile error: https://github.com/fireice-uk/xmr-stak/issues/34
        list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" AND CUDA_VERSION VERSION_GREATER_EQUAL 9.0)
        # workaround find_package(CUDA) is using the wrong path to the CXX host compiler
        # overwrite the CUDA host compiler variable with the used CXX MSVC
        set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
    endif()
    string(REPLACE ";" " " STR_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
    string(APPEND CMAKE_CUDA_FLAGS " ${STR_NVCC_FLAGS}")
    set(CUDA_FOUND TRUE)
else()
    message(FATAL_ERROR "selected CUDA compiler '${CUDA_COMPILER}' is not supported")
    set(CUDA_FOUND FALSE)
endif()
# vim: et sw=4 sts=4 ts=4:
