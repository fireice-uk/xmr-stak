project(xmr-stak)

cmake_minimum_required(VERSION 3.4.0)

# enforce C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# help to find cuda on systems with a software module system
list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")

# help to find AMD OCL SDK Light (replaced APP SDK)
list(APPEND CMAKE_PREFIX_PATH "$ENV{OCL_ROOT}")

# help to find AMD app SDK on systems with a software module system
list(APPEND CMAKE_PREFIX_PATH "$ENV{AMDAPPSDKROOT}")

# allow user to extent CMAKE_PREFIX_PATH via environment variable
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

################################################################################
# CMake user options
################################################################################

# gcc 5.1 is the first GNU version without CoW strings
# https://github.com/fireice-uk/xmr-stak-nvidia/pull/10#issuecomment-290821792
# If you remove this guard to compile with older gcc versions the miner will produce
# a high rate of wrong shares.
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "g++ version must be at least 5.1!")
    endif()
endif()

set(BUILD_TYPE "Release;Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${BUILD_TYPE}")

set(XMR-STAK_COMPILE "native" CACHE STRING "select CPU compute architecture")
set_property(CACHE XMR-STAK_COMPILE PROPERTY STRINGS "native;generic;dev_release")
if(XMR-STAK_COMPILE STREQUAL "native")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS "-march=native -mtune=native ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=native -mtune=native ${CMAKE_C_FLAGS}")
    endif()
elseif(XMR-STAK_COMPILE STREQUAL "generic" OR XMR-STAK_COMPILE STREQUAL "dev_release")
    add_definitions("-DCONF_ENFORCE_OpenCL_1_2=1")
else()
    message(FATAL_ERROR "XMR-STAK_COMPILE is set to an unknown value '${XMR-STAK_COMPILE}'")
endif()

# option to add static libgcc and libstdc++
option(CMAKE_LINK_STATIC "link as much as possible libraries static" OFF)

################################################################################
# Find CUDA
################################################################################
#option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available" FORCE)

option(CUDA_ENABLE "Enable or disable CUDA support (NVIDIA backend)" ON)
if(CUDA_ENABLE)
    find_package(CUDA 7.5)

    if(CUDA_FOUND)
        # required for monero's cryptonight_r
        # libcuda
        find_library(CUDA_LIB 
        NAMES
            libcuda 
            cuda 
            cuda.lib
        HINTS
            ${CUDA_TOOLKIT_ROOT_DIR}
            ${LIBCUDA_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            /usr
            /usr/local/cuda
        PATH_SUFFIXES
            lib64
            lib/x64
            lib/Win32
            lib64/stubs
            lib)

        #nvrtc
        find_library(CUDA_NVRTC_LIB 
        NAMES
            libnvrtc 
            nvrtc 
            nvrtc.lib
        HINTS 
            ${CUDA_TOOLKIT_ROOT_DIR} 
            ${LIBNVRTC_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            /usr 
            /usr/local/cuda
        PATH_SUFFIXES
            lib64
            lib/x64
            lib/Win32
            lib)

        list(APPEND BACKEND_TYPES "nvidia")
        option(XMR-STAK_LARGEGRID "Support large CUDA block count > 128" ON)
        if(XMR-STAK_LARGEGRID)
            add_definitions("-DXMR_STAK_LARGEGRID=${XMR-STAK_LARGEGRID}")
        endif()

        set(DEVICE_COMPILER "nvcc")
        set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

        if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            list(APPEND DEVICE_COMPILER "clang")
        endif()

        set_property(CACHE CUDA_COMPILER PROPERTY STRINGS "${DEVICE_COMPILER}")

        set(XMR-STAK_THREADS 0 CACHE STRING "Set maximum number of threads (for compile time optimization)")
        if(NOT XMR-STAK_THREADS EQUAL 0)
            message(STATUS "xmr-stak-nvidia: set max threads per block to ${XMR-STAK_THREADS}")
            add_definitions("-DXMR_STAK_THREADS=${XMR-STAK_THREADS}")
        endif()
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

        # generate comma separated list with architectures
        string(REPLACE ";" "+" STR_CUDA_ARCH "${CUDA_ARCH}")
        add_definitions("-DXMRSTAK_CUDA_ARCH_LIST=${STR_CUDA_ARCH}")

        # validate architectures (only numbers are allowed)
        foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
            string(REGEX MATCH "^[0-9]+$" IS_NUMBER ${CUDA_ARCH})
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

        option(CUDA_SHOW_REGISTER "Show registers used for each kernel and compute architecture" OFF)
        option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps" OFF)

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
                set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
            endif()

            # required for cryptonight_gpu (fast floating point operations are not allowed)
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --fmad=false --prec-div=true --ftz=false")

            # avoid that nvcc in CUDA 8 complains about sm_20 pending removal
            if(CUDA_VERSION VERSION_EQUAL 8.0)
                set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
            endif()

            # avoid that nvcc in CUDA < 8 tries to use libc `memcpy` within the kernel
            if(CUDA_VERSION VERSION_LESS 8.0)
                add_definitions(-D_FORCE_INLINES)
            endif()
            foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
                # set flags to create device code for the given architecture
                set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
                    "--generate-code arch=compute_${CUDA_ARCH_ELEM},code=sm_${CUDA_ARCH_ELEM} --generate-code arch=compute_${CUDA_ARCH_ELEM},code=compute_${CUDA_ARCH_ELEM}")
            endforeach()

            # give each thread an independent default stream
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")

            option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

            if(CUDA_SHOW_CODELINES)
                set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -lineinfo)
                set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
            endif(CUDA_SHOW_CODELINES)

            if(CUDA_SHOW_REGISTER)
                set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
            endif(CUDA_SHOW_REGISTER)

            if(CUDA_KEEP_FILES)
                set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}")
            endif(CUDA_KEEP_FILES)

            if(CUDA_VERSION VERSION_LESS 8.0)
                # for CUDA 7.5 fix compile error: https://github.com/fireice-uk/xmr-stak/issues/34
                set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-D_MWAITXINTRIN_H_INCLUDED")
            endif()

            # workaround find_package(CUDA) is using the wrong path to the CXX host compiler
            # overwrite the CUDA host compiler variable with the used CXX MSVC
            # in linux where clang and gcc is installed it also helps to select the correct host compiler
            set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)

        else()
            message(FATAL_ERROR "selected CUDA compiler '${CUDA_COMPILER}' is not supported")
        endif()
    else()
        message(FATAL_ERROR "CUDA NOT found: use `-DCUDA_ENABLE=OFF` to build without NVIDIA GPU support")
    endif()
else()
    add_definitions("-DCONF_NO_CUDA")
endif()

###############################################################################
# Find OpenCL
###############################################################################

option(OpenCL_ENABLE "Enable or disable OpenCL spport (AMD GPU support)" ON)
if(OpenCL_ENABLE)
    # try to find AMD OpenCL before NVIDIA OpenCL
    find_path(OpenCL_INCLUDE_DIR
        NAMES
            CL/cl.h
            OpenCL/cl.h
        NO_DEFAULT_PATH
        PATHS
            ENV "OCL_ROOT"
            ENV "OpenCL_ROOT"
            ENV AMDAPPSDKROOT
            ENV ATISTREAMSDKROOT
            ENV "PROGRAMFILES(X86)"
            /usr
        PATH_SUFFIXES
            include
            OpenCL/common/inc
            "AMD APP/include")

    find_library(OpenCL_LIBRARY
        NAMES
            OpenCL
            OpenCL.lib
        NO_DEFAULT_PATH
        PATHS
            ENV "OCL_ROOT"
            ENV "OpenCL_ROOT"
            ENV AMDAPPSDKROOT
            ENV ATISTREAMSDKROOT
            ENV "PROGRAMFILES(X86)"
        PATH_SUFFIXES
            "AMD APP/lib/x86_64"
            lib/x86_64
            lib/x86_64-linux-gnu
            lib/x64
            OpenCL/common/lib/x64)
    # find package will use the previews searched path variables
    find_package(OpenCL)
    if(OpenCL_FOUND)
        list(APPEND BACKEND_TYPES "amd")
        include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
        #set(LIBS ${LIBS} ${OpenCL_LIBRARY})
        link_directories(${OpenCL_LIBRARY})
    else()
        message(FATAL_ERROR "OpenCL NOT found: use `-DOpenCL_ENABLE=OFF` to build without OpenCL support for AMD gpu's")
    endif()
else()
    add_definitions("-DCONF_NO_OPENCL")
endif()

###############################################################################
# CPU backend
###############################################################################

option(CPU_ENABLE "Enable or disable CPU support" ON)
if(NOT CPU_ENABLE)
    add_definitions("-DCONF_NO_CPU")
else()
    list(APPEND BACKEND_TYPES "cpu")
endif()

################################################################################
# Explicit march setting for Clang
################################################################################

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties(xmrstak/backend/cpu/crypto/cn_gpu_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
endif()

################################################################################
# Find PThreads
################################################################################

if(NOT WIN32)
    find_package(Threads REQUIRED)
    set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()
################################################################################
# Find microhttpd
################################################################################

option(MICROHTTPD_ENABLE "Enable or disable the requirement of microhttp (http daemon)" ON)
if(MICROHTTPD_ENABLE)
    find_path(MTHD_INCLUDE_DIR
        NAMES
            microhttpd.h
        PATHS
            /opt/local
            /usr/local
            /usr
            ENV "PROGRAMFILES(X86)"
            ENV "MICROHTTPD_ROOT"
        PATH_SUFFIXES
            include)

    find_library(MHTD
        NAMES
            microhttpd
            libmicrohttpd.lib
        PATHS
            ENV "MICROHTTPD_ROOT"
        PATH_SUFFIXES
            lib)
    if(MHTD STREQUAL "MHTD-NOTFOUND")
        message(FATAL_ERROR "microhttpd NOT found: use `-DMICROHTTPD_ENABLE=OFF` to build without http daemon support")
    else()
        set(LIBS ${LIBS} ${MHTD})
        include_directories(AFTER ${MTHD_INCLUDE_DIR})
    endif()
else()
    add_definitions("-DCONF_NO_HTTPD")
endif()

###############################################################################
# Find OpenSSL
###############################################################################

option(OpenSSL_ENABLE "Enable or disable the requirement of OpenSSL" ON)
if(OpenSSL_ENABLE)
    find_package(OpenSSL)
    if(OPENSSL_FOUND)
        include_directories(${OPENSSL_INCLUDE_DIR})
        set(LIBS ${LIBS} ${OPENSSL_LIBRARIES})
    else()
        message(FATAL_ERROR "OpenSSL NOT found: use `-DOpenSSL_ENABLE=OFF` to build without SSL support")
    endif()
else()
    add_definitions("-DCONF_NO_TLS")
endif()

################################################################################
# Find hwloc
################################################################################

option(HWLOC_ENABLE "Enable or disable the requirement of hwloc" ON)
if(HWLOC_ENABLE)
    find_path(HWLOC_INCLUDE_DIR
        NAMES
            hwloc.h
        PATHS
            /opt/local
            /usr/local
            /usr
            ENV "PROGRAMFILES(X86)"
            ENV "HWLOC_ROOT"
        PATH_SUFFIXES
            include)

    find_library(HWLOC
        NAMES
            libhwloc.lib
            hwloc
        PATHS
            ENV "HWLOC_ROOT"
        PATH_SUFFIXES
            lib)

    if(HWLOC STREQUAL "HWLOC-NOTFOUND" OR ${HWLOC_INCLUDE_DIR} STREQUAL "HWLOC_INCLUDE_DIR-NOTFOUND")
        message(FATAL_ERROR "hwloc NOT found: use `-DHWLOC_ENABLE=OFF` to build without hwloc support")
    else()
        set(LIBS ${LIBS} ${HWLOC})
        include_directories(AFTER ${HWLOC_INCLUDE_DIR})
    endif()
else()
    add_definitions("-DCONF_NO_HWLOC")
endif()

################################################################################
# Windows Sockets
################################################################################

if(WIN32)
    set(LIBS ${LIBS} wsock32 ws2_32)
endif()

################################################################################
# Versioning
################################################################################

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT GIT_COMMIT_HASH STREQUAL "")
    add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
endif()
if(NOT GIT_BRANCH STREQUAL "")
    add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")
endif()

# generate backend string
string(REPLACE ";" "-" STR_BACKEND_TYPES "${BACKEND_TYPES}")
add_definitions("-DBACKEND_TYPE=${STR_BACKEND_TYPES}")

################################################################################
# Compile & Link
################################################################################

include_directories(BEFORE .)

set (CMAKE_POSITION_INDEPENDENT_CODE TRUE)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # remove warnings that f_open() is not save and f_open_s should be used
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    # disable min define to allow usage of std::min
    add_definitions(-DNOMINMAX)
else()
    # activate sse2 and aes-ni
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -maes")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2 -maes")
endif()

# add -Wall for debug builds with gcc
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    endif()
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-Wl,-z,noexecstack ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-Wl,-z,noexecstack ${CMAKE_C_FLAGS}")
endif()

if(XMR-STAK_COMPILE STREQUAL "dev_release")
    add_definitions(-DXMRSTAK_DEV_RELEASE)
endif()

# activate static libgcc and libstdc++ linking
if(CMAKE_LINK_STATIC)
    set(BUILD_SHARED_LIBRARIES OFF)
    set(DL_LIB ${CMAKE_DL_LIBS})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
        set(LIBS "-static-libgcc -static-libstdc++ ${LIBS}")
    endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # asm optimized monero v8 code
    enable_language(ASM_MASM)
    set_property(SOURCE "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.asm" PROPERTY ASM_MASM)
    set_property(SOURCE "xmrstak/backend/cpu/crypto/asm/cnR/CryptonightR_template.asm" PROPERTY ASM_MASM)
    add_library(xmr-stak-asm
        STATIC
        "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.asm"
        "xmrstak/backend/cpu/crypto/asm/cnR/CryptonightR_template.asm"
    )
else()
    # asm optimized monero v8 code
    enable_language(ASM)
    set_property(SOURCE "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.S" PROPERTY CPP)
    set_property(SOURCE "xmrstak/backend/cpu/crypto/asm/cnR/CryptonightR_template.S" PROPERTY CPP)
    set_source_files_properties("xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    set_source_files_properties("xmrstak/backend/cpu/crypto/asm/cnR/CryptonightR_template.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    add_library(xmr-stak-asm
        STATIC
        "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.S"
        "xmrstak/backend/cpu/crypto/asm/cnR/CryptonightR_template.S"
    )
endif()

set_property(TARGET xmr-stak-asm PROPERTY LINKER_LANGUAGE C)

# compile C files
file(GLOB SRCFILES_C "xmrstak/backend/cpu/crypto/*.c")

add_library(xmr-stak-c
    STATIC
    ${SRCFILES_C}
)
set_property(TARGET xmr-stak-c PROPERTY C_STANDARD 99)
if(MICROHTTPD_ENABLE)
    target_link_libraries(xmr-stak-c ${MHTD})
endif()
target_link_libraries(xmr-stak-c ${LIBS} xmr-stak-asm)

# compile generic backend files
file(GLOB BACKEND_CPP
    "xmrstak/*.cpp"
    "xmrstak/backend/cpu/*.cpp"
    "xmrstak/backend/*.cpp"
    "xmrstak/backend/cpu/crypto/*.cpp"
    "xmrstak/http/*.cpp"
    "xmrstak/misc/*.cpp"
    "xmrstak/net/*.cpp")

add_library(xmr-stak-backend
    STATIC
    ${BACKEND_CPP}
)
target_link_libraries(xmr-stak-backend xmr-stak-c ${CMAKE_DL_LIBS} xmr-stak-asm)

# compile CUDA backend
if(CUDA_FOUND)
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

    set(CUDA_LIBRARIES ${CUDA_LIB} ${CUDA_LIBRARIES})
    if(XMR-STAK_COMPILE STREQUAL "dev_release")
        # do not link nvrtc for linux binaries, cn-r will be disabled
        if(WIN32)
            set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB})
        endif()
    else()
        set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB})
    endif()
    
    target_link_libraries(xmrstak_cuda_backend ${CUDA_LIBRARIES})
    target_link_libraries(xmrstak_cuda_backend xmr-stak-backend xmr-stak-asm)
endif()

# compile AMD backend
if(OpenCL_FOUND)
    file(GLOB OPENCLSRCFILES
        "xmrstak/backend/amd/amd_gpu/*.cpp"
        "xmrstak/backend/amd/*.cpp")
    add_library(xmrstak_opencl_backend
        SHARED
        ${OPENCLSRCFILES}
    )
    target_link_libraries(xmrstak_opencl_backend ${OpenCL_LIBRARY} )
    target_link_libraries(xmrstak_opencl_backend xmr-stak-backend xmr-stak-asm)
endif()

# compile final binary
file(GLOB SRCFILES_CPP "xmrstak/cli/*.cpp")
set_source_files_properties(${SRCFILES_CPP} PROPERTIES LANGUAGE CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_executable(xmr-stak ${SRCFILES_CPP} xmrstak/cli/xmr-stak.manifest)
else()
    add_executable(xmr-stak ${SRCFILES_CPP})
endif()

set(EXECUTABLE_OUTPUT_PATH "bin" CACHE STRING "Path to place executables relative to ${CMAKE_INSTALL_PREFIX}")
set(LIBRARY_OUTPUT_PATH "bin" CACHE STRING "Path to place libraries relative to ${CMAKE_INSTALL_PREFIX}")

target_link_libraries(xmr-stak ${LIBS} xmr-stak-c xmr-stak-backend xmr-stak-asm)

################################################################################
# Install
################################################################################

# do not install the binary if the project and install are equal
if( NOT CMAKE_INSTALL_PREFIX STREQUAL PROJECT_BINARY_DIR )
    install(TARGETS xmr-stak
            RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${EXECUTABLE_OUTPUT_PATH}")
    if(CUDA_FOUND)
        if(WIN32)
            install(TARGETS xmrstak_cuda_backend
                RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${LIBRARY_OUTPUT_PATH}")
        else()
            install(TARGETS xmrstak_cuda_backend
                LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/${LIBRARY_OUTPUT_PATH}")
        endif()
    endif()
    if(OpenCL_FOUND)
        if(WIN32)
            install(TARGETS xmrstak_opencl_backend
                RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${LIBRARY_OUTPUT_PATH}")
        else()
            install(TARGETS xmrstak_opencl_backend
                LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/${LIBRARY_OUTPUT_PATH}")
        endif()
    endif()
else()
    # this rule is used if the install prefix is the build directory
    install(CODE "MESSAGE(\"xmr-stak installed to folder 'bin'\")")
endif()
