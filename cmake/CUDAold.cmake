################################################################################
# Define backend: nvidia
################################################################################
#option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available" FORCE)
option(CUDA_ENABLE "Enable or disable CUDA support (NVIDIA backend)" ON)
if(CUDA_ENABLE)
    # Find CUDA
    # help for systems with a software module system
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")

    find_package(CUDA 7.5)
    if(CUDA_FOUND)
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "All")
        message(STATUS "ARCH_FLAGS: ${ARCH_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
        foreach(THIS_FLAG ${ARCH_FLAGS})
            message(STATUS "THIS_FLAG: ${THIS_FLAG}")
            string(REGEX MATCH "^arch[0-9]+$" IS_ARCH ${THIS_FLAG})
            if(IS_ARCH)
                list(APPEND DEFAULT_CUDA_ARCH ${THIS_FLAG})
            endif()
        endforeach()
        #message(STATUS "ARCHS: ${ARCH_FLAGS}")

        option(CUDA_LARGEGRID "Support large CUDA block count > 128" ON)
        set(CUDA_THREADS_MAX 0 CACHE STRING "Set maximum number of threads (for compile time optimization)")
        option(CUDA_SHOW_REGISTER "Show registers used for each kernel and compute architecture" OFF)
        option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps" OFF)
        option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

        set(DEVICE_COMPILER "nvcc")
        set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

        if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            list(APPEND DEVICE_COMPILER "clang")
        endif()

        set_property(CACHE CUDA_COMPILER PROPERTY STRINGS "${DEVICE_COMPILER}")

        set(DEFAULT_CUDA_ARCH "30;35;37;50;52")
        # Fermi GPUs are only supported with CUDA < 9.0
        if(CUDA_VERSION VERSION_LESS 9.0)
            list(APPEND DEFAULT_CUDA_ARCH "20")
        endif()
        # add Pascal support for CUDA >= 8.0
        if(NOT CUDA_VERSION VERSION_LESS 8.0)
            list(APPEND DEFAULT_CUDA_ARCH "60" "61" "62")
        endif()
        # add Volta support for CUDA >= 9.0
        if(NOT CUDA_VERSION VERSION_LESS 9.0)
            # Volta GPUs are currently not supported on MACOSX
            # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-general-known-issues
            if(NOT APPLE)
                list(APPEND DEFAULT_CUDA_ARCH "70")
            endif()
        endif()
        # add Turing support for CUDA >= 10.0
        if(NOT CUDA_VERSION VERSION_LESS 10.0)
            list(APPEND DEFAULT_CUDA_ARCH "75")
        endif()
        set(CUDA_ARCH "${DEFAULT_CUDA_ARCH}" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")
        list(SORT CUDA_ARCH)
        list(REMOVE_DUPLICATES CUDA_ARCH)

        # validate architectures (only numbers are allowed)
        foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
            string(REGEX MATCH "^[0-9]+$" IS_NUMBER ${CUDA_ARCH_ELEM})
            if(NOT IS_NUMBER)
                message(FATAL_ERROR "Defined compute architecture '${CUDA_ARCH_ELEM}' in "
                                    "'${CUDA_ARCH}' is not an integral number, use e.g. '30' (for compute architecture 3.0).")
            endif()
            unset(IS_NUMBER)

            if(${CUDA_ARCH_ELEM} LESS 20)
                message(FATAL_ERROR "Unsupported CUDA architecture '${CUDA_ARCH_ELEM}' specified. "
                                    "Use '20' (for compute architecture 2.0) or higher.")
            endif()
        endforeach()

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
        elseif(CUDA_COMPILER STREQUAL "nvcc")
            # add c++11 for cuda
            if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
                list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
            endif()

            # avoid that nvcc in CUDA 8 complains about sm_20 pending removal
            if(CUDA_VERSION VERSION_EQUAL 8.0)
                list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
            endif()

            foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
                # set flags to create device code for the given architecture
                list(APPEND CUDA_NVCC_FLAGS
                    "--generate-code arch=compute_${CUDA_ARCH_ELEM},code=sm_${CUDA_ARCH_ELEM}"
                    "--generate-code arch=compute_${CUDA_ARCH_ELEM},code=compute_${CUDA_ARCH_ELEM}")
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
        else()
            message(FATAL_ERROR "selected CUDA compiler '${CUDA_COMPILER}' is not supported")
        endif()

        ###############################################################################
        # Define target: xmrstak_cuda_backend; nvidia backend shared lib
        ###############################################################################
        file(GLOB CUDASRCFILES
            "xmrstak/backend/nvidia/nvcc_code/*.cu"
            "xmrstak/backend/nvidia/*.cpp")
        if(CUDA_COMPILER STREQUAL "clang")
            # build device code with clang
            add_library(
                xmrstak_cuda_backend
                SHARED
                ${CUDASRCFILES}
            )
            set_target_properties(xmrstak_cuda_backend PROPERTIES COMPILE_FLAGS ${CLANG_BUILD_FLAGS})
            set_target_properties(xmrstak_cuda_backend PROPERTIES LINKER_LANGUAGE CXX)
            set_source_files_properties(${CUDASRCFILES} PROPERTIES LANGUAGE CXX)
        else()
            #  build device code with nvcc
            cuda_add_library(
                xmrstak_cuda_backend
                SHARED
                ${CUDASRCFILES}
            )
        endif()

        if(NOT CUDA_THREADS_MAX EQUAL 0)
            message(STATUS "xmr-stak-nvidia: set max threads per block to ${CUDA_THREADS_MAX}")
            target_compile_definitions(xmrstak_cuda_backend PUBLIC "CUDA_THREADS_MAX=${CUDA_THREADS_MAX}")
        endif()

        # generate comma separated list with architectures
        string(REPLACE ";" "+" STR_CUDA_ARCH "${CUDA_ARCH}")
        target_compile_definitions(xmrstak_cuda_backend PUBLIC "XMRSTAK_CUDA_ARCH_LIST=${STR_CUDA_ARCH}")

        target_link_libraries(xmrstak_cuda_backend ${CUDA_LIBRARIES} xmr-stak-backend)
        list(APPEND BACKEND_TYPES "nvidia")
    else()
        message(FATAL_ERROR "CUDA NOT found: use `-DCUDA_ENABLE=OFF` to build without NVIDIA GPU support")
    endif()
else()
    target_compile_definitions(xmr-stak-c PUBLIC CONF_NO_CUDA)
endif()
# vim: et sw=4 sts=4 ts=4:
