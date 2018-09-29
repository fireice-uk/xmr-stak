cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
message("* COMPILERS: detect and configure...")
project(xmr-stak)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(funcUtil)

# enforce C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

################################################################################
# CMake user options
################################################################################
option(HWLOC_ENABLE "Enable or disable the requirement of hwloc" ON)
option(OpenSSL_ENABLE "Enable or disable the requirement of OpenSSL" ON)
option(MICROHTTPD_ENABLE "Enable or disable the requirement of microhttpd (http daemon)" ON)
option(CPU_ENABLE "Enable or disable CPU support" ON)
option(CUDA_ENABLE "Enable or disable CUDA support (NVIDIA backend)" ON)
option(OpenCL_ENABLE "Enable or disable OpenCL support (AMD GPU support)" ON)
option(CMAKE_LINK_STATIC "link as much as possible libraries static" OFF)

set(EXECUTABLE_OUTPUT_PATH "bin" CACHE STRING "Path to place executables relative to ${CMAKE_INSTALL_PREFIX}")
set(LIBRARY_OUTPUT_PATH "bin" CACHE STRING "Path to place libraries relative to ${CMAKE_INSTALL_PREFIX}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# allow user to extend CMAKE_PREFIX_PATH via environment variable
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

set(BUILD_TYPE "Release;Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${BUILD_TYPE}")

set(XMR-STAK_COMPILE "native" CACHE STRING "select CPU compute architecture")
set_property(CACHE XMR-STAK_COMPILE PROPERTY STRINGS "native;generic")
if(NOT (XMR-STAK_COMPILE STREQUAL "generic" OR XMR-STAK_COMPILE STREQUAL "native"))
    message(FATAL_ERROR "XMR-STAK_COMPILE is set to an unknown value '${XMR-STAK_COMPILE}'")
endif()

# option to add static libgcc and libstdc++
if(CMAKE_LINK_STATIC)
    # activate static libgcc and libstdc++ linking
    set(BUILD_SHARED_LIBRARIES OFF)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
    #set(CMAKE_FIND_LIBRARY_SUFFIXES .a .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
else()
    set(BUILD_SHARED_LIBRARIES ON)
    set(OPENSSL_USE_STATIC_LIBS FALSE)
    set(MICROHTTPD_USE_STATIC_LIBS FALSE)
endif()

################################################################################
# Find Threads (prefer Pthreads and compiler native flag)
################################################################################
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG FALSE)
find_package(Threads REQUIRED QUIET)

################################################################################
# Configure Assembler
################################################################################
message("* COMPILERS(ASM): detect and configure...")
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    enable_language(ASM_MASM)
else()
    enable_language(ASM)
endif()
message("* COMPILERS(ASM): configuration complete!")

################################################################################
# Configure NVCC
################################################################################
if(CUDA_ENABLE)
    option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available" FORCE)
    option(CUDA_SHOW_REGISTER "Show registers used for each kernel and compute architecture" OFF)
    option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps" OFF)
    option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
    option(CUDA_LARGEGRID "Support large CUDA block count > 128" ON)
    set(CUDA_THREADS_MAX 0 CACHE STRING "Set maximum number of threads (for compile time optimization)")
    set(CUDA_ARCH_USER "${CUDA_ARCH}" CACHE STRING "User supplied CUDA_ARCH before filtering")
    include(funcCUDA)

    message("* COMPILERS(CUDA): detect and configure...")
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" CUDA_PATH)
    if("" STREQUAL "${CUDA_PATH}")
        #try normal system paths
        find_program(_nvccbin NAME "nvcc")
    else()
        #try envvar CUDA_PATH first
        set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_PATH}")
        find_program(_nvccbin NAME "nvcc"
                PATHS "${CUDA_PATH}" "${CMAKE_SYSTEM_PROGRAM_PATH}"
                PATH_SUFFIXES "bin" ""
                NO_DEFAULT_PATH)
    endif()
    if(NOT "${_nvccbin}" MATCHES "NOTFOUND")
        set(CUDACXX "${_nvccbin}")
        set(CMAKE_CUDA_COMPILER "${_nvccbin}")
        message(STATUS "Found nvcc: ${_nvccbin}")
    endif()
    find_package(CUDA REQUIRED QUIET)
    enable_language(CUDA)
    cuda_arch_probe()
    message(STATUS "The CUDA compiler architectures are ${CMAKE_CUDA_ARCHLIST}")
    if("" STREQUAL "${CUDA_ARCH}")
        message(STATUS "Provided CUDA_ARCH empty, build for ALL (this will be large)")
        message(STATUS "You should set CUDA_ARCH to a subset of the above")
        set(CUDA_ARCH_OUT "${CMAKE_CUDA_ARCHLIST}")
    else()
        cuda_arch_filter("${CUDA_ARCH}" "${CMAKE_CUDA_ARCHLIST}" CUDA_ARCH_FILTERED)
        if("" STREQUAL "${CUDA_ARCH_FILTERED}")
            message(FATAL_ERROR "Provided CUDA_ARCH failed, note the list above")
        endif()
        set(CUDA_ARCH_OUT "${CUDA_ARCH_FILTERED}")
        unset(CUDA_ARCH_FILTERED)
    endif()
    set(CUDA_ARCH "${CUDA_ARCH_OUT}" CACHE STRING
            "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=${CMAKE_CUDA_ARCHLIST}')" FORCE)
    unset(CUDA_ARCH_OUT)
    message("* COMPILERS(CUDA): configuration complete!")
endif()
message("* COMPILERS: configuration complete!")

###############################################################################
# Define target: xmr-stak; Main CLI
###############################################################################
set(SRCFILES_CPP "xmrstak/cli/cli-miner.cpp")
set_source_files_properties(${SRCFILES_CPP} PROPERTIES LANGUAGE CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    list(APPEND SRCFILES_CPP "xmrstak/cli/xmr-stak.manifest")
endif()
add_executable(xmr-stak ${SRCFILES_CPP})

###############################################################################
# Define target: xmr-stak-c; Crypto Library
###############################################################################
add_library(xmr-stak-c
    STATIC
    "xmrstak/backend/cpu/crypto/c_blake256.c"
    "xmrstak/backend/cpu/crypto/c_groestl.c"
    "xmrstak/backend/cpu/crypto/c_jh.c"
    "xmrstak/backend/cpu/crypto/c_keccak.c"
    "xmrstak/backend/cpu/crypto/c_skein.c"
)
set_property(TARGET xmr-stak-c PROPERTY C_STANDARD 99)
#target_link_libraries(xmr-stak-c Threads::Threads)

###############################################################################
# Define target: xmr-stak-asm; Crypto Library, optional assembler code
###############################################################################
if(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # asm optimized monero v8 code
    set_property(SOURCE "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.asm" PROPERTY ASM_MASM)
    add_library(xmr-stak-asm
        STATIC
        "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.asm"
    )
else()
    # default dialect with C-preprocessor pass
    set_property(SOURCE "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.S" PROPERTY CPP)
    set_source_files_properties("xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    add_library(xmr-stak-asm
        STATIC
        "xmrstak/backend/cpu/crypto/asm/cryptonight_v8_main_loop.S"
    )
endif()
set_property(TARGET xmr-stak-asm PROPERTY LINKER_LANGUAGE C)
target_link_libraries(xmr-stak-c xmr-stak-asm)

###############################################################################
# Define target: xmr-stak-backend; generic backend files
###############################################################################
add_library(xmr-stak-backend
    STATIC
    "xmrstak/version.cpp"
    "xmrstak/jconf.cpp"
    "xmrstak/backend/cpu/jconf.cpp"
    "xmrstak/backend/cpu/cpuType.cpp"
    "xmrstak/backend/cpu/hwlocMemory.cpp"
    "xmrstak/backend/cpu/minethd.cpp"
    "xmrstak/backend/backendConnector.cpp"
    "xmrstak/backend/globalStates.cpp"
    "xmrstak/backend/cpu/crypto/cryptonight_common.cpp"
    "xmrstak/http/webdesign.cpp"
    "xmrstak/http/httpd.cpp"
    "xmrstak/misc/console.cpp"
    "xmrstak/misc/executor.cpp"
    "xmrstak/misc/telemetry.cpp"
    "xmrstak/misc/uac.cpp"
    "xmrstak/misc/utility.cpp"
    "xmrstak/net/jpsock.cpp"
    "xmrstak/net/socket.cpp"
)
target_link_libraries(xmr-stak-backend xmr-stak-c)

################################################################################
# Probe compilers, work around bugs, add optimal flags
################################################################################
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # remove warnings that f_open() is not save and f_open_s should be used
    target_compile_definitions(xmr-stak-c PUBLIC _CRT_SECURE_NO_DEPRECATE)
    # disable min define to allow usage of std::min
    target_compile_definitions(xmr-stak-c PUBLIC NOMINMAX)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    get_filename_component(ICC_ROOT ${CMAKE_C_COMPILER} DIRECTORY)
    get_filename_component(ICC_ROOT ${ICC_ROOT} DIRECTORY)
    message(STATUS "ICC: adding include+lib root ${ICC_ROOT}")
    list(INSERT 0 CMAKE_PREFIX_PATH "${ICC_ROOT}")
    #include_directories("${ICC_ROOT}/include")
    link_directories("${ICC_ROOT}/lib")
    unset(ICC_REAL)
    unset(ICC_ROOT)
    if(NOT WIN32)
        execute_process(
            COMMAND gcc -dumpversion
            OUTPUT_VARIABLE GCCMAJOR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 17) AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19))
            #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
            #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
            # locate best gcc version to pass to icc-18 (gcc-8 doesn't work, for example)
            if(${GCCMAJOR} VERSION_GREATER 7 OR DEFINED ICC_GCCVER)
                # Use -DICC_GCCVER=6 for example to force 'gcc-6' and 'g++-6'
                if(DEFINED ICC_GCCVER)
                    # just literally override, no checks
                    set(ICC_GXX "g++-${ICC_GCCVER}")
                    set(ICC_GCC "gcc-${ICC_GCCVER}")
                else()
                    message(STATUS "ICC: version 18 can't use system compiler gcc-${GCCMAJOR}, seeking alternative")
                    # Otherwise, the newest available and compatible pair will be selected
                    find_program(ICC_GXX NAMES "g++-7" "g++-6" "g++")
                    get_filename_component(ICC_GXX ${ICC_GXX} NAME)
                    find_program(ICC_GCC NAMES "gcc-7" "gcc-6" "gcc")
                    get_filename_component(ICC_GCC ${ICC_GCC} NAME)
                endif()
                message(STATUS "ICC: setting subcompiler flags: -gxx-name=${ICC_GXX} -gcc-name=${ICC_GCC}")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gxx-name=${ICC_GXX} -gcc-name=${ICC_GCC}")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gxx-name=${ICC_GXX} -gcc-name=${ICC_GCC}")
                unset(ICC_GXX)
                unset(ICC_GCC)
            endif()
        endif()
    endif()
    if(XMR-STAK_COMPILE STREQUAL "native")
        # activate optimize for current host CPU
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fast")
        target_compile_definitions(xmr-stak-c PUBLIC __PURE_INTEL_C99_HEADERS__)
    endif()
    if(CMAKE_LINK_STATIC)
        set(LIBS "-static-libgcc -static-libstdc++ -static-intel ${LIBS}")
    endif()
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
            # gcc 5.1 is the first GNU version without CoW strings
            # https://github.com/fireice-uk/xmr-stak-nvidia/pull/10#issuecomment-290821792
            # If you remove this guard to compile with older gcc versions the miner will produce
            # a high rate of wrong shares.
            message(FATAL_ERROR "g++ version must be at least 5.1!")
        endif()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	        # add -Wall for debug builds with gcc
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
        endif()
    endif()
    if(CMAKE_LINK_STATIC AND (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
        set(LIBS "-static-libgcc -static-libstdc++ ${LIBS}")
    endif()
    # below here is applied to all other compilers including GNU
    if(XMR-STAK_COMPILE STREQUAL "native")
        # activate optimize for current host CPU
        set(CMAKE_CXX_FLAGS "-march=native -mtune=native ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=native -mtune=native ${CMAKE_C_FLAGS}")
    endif()
    # activate sse2 and aes-ni in all cases
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -maes")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2 -maes")
endif()

#########################
### LIBRARIES SECTION ###
#########################
message("* LIBS: detect and configure...")
if(CMAKE_LINK_STATIC)
    message(STATUS "Library mode: STATIC")
else()
    message(STATUS "Library mode: SHARED")
endif()
################################################################################
# Define feature: Windows Sockets
################################################################################
if(WIN32)
    message(STATUS "Adding Windows Socket Libraries: wsock32 ws2_32")
    target_link_libraries(xmr-stak-backend wsock32 ws2_32)
endif()

################################################################################
# Define feature: libhwloc
################################################################################
if(HWLOC_ENABLE)
    # Find hwloc
    find_path(HWLOC_INCLUDE_DIR
        NAMES
            hwloc.h
        PATHS
            ENV "HWLOC_ROOT"
            /opt/local
            /usr/local
            /usr
            ENV "PROGRAMFILES(X86)"
        PATH_SUFFIXES
            include)

    find_library(HWLOC_LIBRARY
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
        set(HWLOC_FOUND TRUE)
        listFilterLibs("${HWLOC_LIBRARY}" HWLOC_LIBRARY)
        message(STATUS "Found hwloc @ ${HWLOC_LIBRARY} (${HWLOC_INCLUDE_DIR})")
        add_library(libhwloc MODULE IMPORTED)
        target_include_directories(libhwloc INTERFACE "${HWLOC_INCLUDE_DIR}")
        set_target_properties(libhwloc PROPERTIES IMPORTED_LOCATION "${HWLOC_LIBRARY}")
        target_link_libraries(xmr-stak-backend libhwloc)
    endif()
else()
    target_compile_definitions(xmr-stak-backend PUBLIC CONF_NO_HWLOC)
endif()

###############################################################################
# Define feature: OpenSSL (for connection to pools)
###############################################################################
if(OpenSSL_ENABLE)
    # Find OpenSSL
    find_package(OpenSSL QUIET)
    if(OPENSSL_FOUND)
        if(OPENSSL_SSL_LIBRARY)
            listFilterLibs("${OPENSSL_SSL_LIBRARY}" OPENSSL_SSL_LIBRARY)
            message(STATUS "Found OpenSSL v${OPENSSL_VERSION} libssl @ ${OPENSSL_SSL_LIBRARY} (${OPENSSL_INCLUDE_DIR})")
            add_library(libssl MODULE IMPORTED)
            target_include_directories(libssl INTERFACE "${OPENSSL_INCLUDE_DIR}")
            set_target_properties(libssl PROPERTIES IMPORTED_LOCATION "${OPENSSL_SSL_LIBRARY}")
            target_link_libraries(xmr-stak-backend libssl)
        endif()
        if(OPENSSL_CRYPTO_LIBRARY)
            listFilterLibs("${OPENSSL_CRYPTO_LIBRARY}" OPENSSL_CRYPTO_LIBRARY)
            message(STATUS "Found OpenSSL v${OPENSSL_VERSION} libcrypto @ ${OPENSSL_CRYPTO_LIBRARY} (${OPENSSL_INCLUDE_DIR})")
            add_library(libcrypto MODULE IMPORTED)
            target_include_directories(libcrypto INTERFACE "${OPENSSL_INCLUDE_DIR}")
            set_target_properties(libcrypto PROPERTIES IMPORTED_LOCATION "${OPENSSL_CRYPTO_LIBRARY}")
            target_link_libraries(xmr-stak-backend libcrypto)
        endif()
        target_link_libraries(xmr-stak-backend ${CMAKE_DL_LIBS})
    else()
        message(FATAL_ERROR "OpenSSL NOT found: use `-DOpenSSL_ENABLE=OFF` to build without SSL support")
    endif()
else()
    target_compile_definitions(xmr-stak-backend PUBLIC CONF_NO_TLS)
endif()

################################################################################
# Define feature: microhttpd
################################################################################
if(MICROHTTPD_ENABLE)
    find_package(MicroHttpd QUIET)
    if(MICROHTTPD_FOUND)
        listFilterLibs("${MICROHTTPD_LIBRARY}" MICROHTTPD_LIBRARY)
        message(STATUS "Found microhttpd v${MICROHTTPD_VERSION} @ ${MICROHTTPD_LIBRARY} (${MICROHTTPD_INCLUDE_DIR})")
        add_library(libmicrohttpd MODULE IMPORTED)
        target_include_directories(libmicrohttpd INTERFACE "${MICROHTTPD_INCLUDE_DIR}")
        set_target_properties(libmicrohttpd PROPERTIES IMPORTED_LOCATION "${MICROHTTPD_LIBRARY}")
        target_link_libraries(xmr-stak-backend libmicrohttpd "${MICROHTTPD_LIBS_EXTRA}")
        target_link_libraries(xmr-stak "${MICROHTTPD_LIBS_EXTRA}")
    else()
        message(FATAL_ERROR "microhttpd NOT found: use `-DMICROHTTPD_ENABLE=OFF` to build without http daemon support")
    endif()
else()
    target_compile_definitions(xmr-stak-backend PUBLIC CONF_NO_HTTPD)
endif()
message("* LIBS configuration complete!")

#########################
### BACKENDS SECTION ###
#########################
message("* BACKENDS: detect and configure...")
###############################################################################
# Define backend: cpu
###############################################################################
if(NOT CPU_ENABLE)
    target_compile_definitions(xmr-stak-backend PUBLIC CONF_NO_CPU)
    message(STATUS "BACKENDS: Disable CPU")
else()
    list(APPEND BACKEND_TYPES "cpu")
    message(STATUS "BACKENDS: Enable CPU")
endif()


################################################################################
# Define backend: nvidia
################################################################################
if(CUDA_ENABLE)
    if(CMAKE_VERSION VERSION_LESS 3.8)
        include(CUDAold)
    else()
        include(CUDA)
    endif()
    if(CUDA_FOUND)
        ###############################################################################
        # Define target: xmrstak_cuda_backend; nvidia backend shared lib
        ###############################################################################
        set(CUDASRCFILES "")
        list(APPEND CUDASRCFILES
            "xmrstak/backend/nvidia/jconf.cpp"
            "xmrstak/backend/nvidia/nvcc_code/cuda_core.cu"
            "xmrstak/backend/nvidia/nvcc_code/cuda_extra.cu"
            "xmrstak/backend/nvidia/minethd.cpp"
        )
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR CUDA_COMPILER STREQUAL "nvcc")
            # build device code with nvcc
            if(NOT CMAKE_VERSION VERSION_LESS 3.8)
                #native CUDA support
                add_library(
                    xmrstak_cuda_backend
                    SHARED
                    ${CUDASRCFILES}
                )
                #set_target_properties(xmrstak_cuda_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                set_target_properties(xmrstak_cuda_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
            else()
                #plugin CUDA support (old)
                cuda_add_library(
                    xmrstak_cuda_backend
                    SHARED
                    ${CUDASRCFILES}
                )
            endif()
        else()
            # build device code with clang
            add_library(
                xmrstak_cuda_backend
                SHARED
                ${CUDASRCFILES}
            )
            set_target_properties(xmrstak_cuda_backend PROPERTIES COMPILE_FLAGS ${CLANG_BUILD_FLAGS})
            set_target_properties(xmrstak_cuda_backend PROPERTIES LINKER_LANGUAGE CXX)
            set_source_files_properties(${CUDASRCFILES} PROPERTIES LANGUAGE CXX)
        endif()
        if(NOT CUDA_THREADS_MAX EQUAL 0)
            message(STATUS "xmr-stak-nvidia: set max threads per block to ${CUDA_THREADS_MAX}")
            target_compile_definitions(xmrstak_cuda_backend PUBLIC "CUDA_THREADS_MAX=${CUDA_THREADS_MAX}")
        endif()

        # generate comma separated list with architectures
        string(REPLACE ";" "+" STR_CUDA_ARCH "${CUDA_ARCH}")
        target_compile_definitions(xmr-stak-backend PUBLIC "XMRSTAK_CUDA_ARCH_LIST=${STR_CUDA_ARCH}")

        target_link_libraries(xmrstak_cuda_backend xmr-stak-backend ${CUDA_LIBRARIES})
        list(APPEND BACKEND_TYPES "nvidia")
        message(STATUS "BACKENDS: Enable CUDA"
            " v${CUDA_VERSION}"
            " @ ${CUDA_TOOLKIT_ROOT_DIR}"
        )
    else()
        message(FATAL_ERROR "CUDA NOT found: use `-DCUDA_ENABLE=OFF` to build without NVIDIA GPU support")
    endif()
else()
    target_compile_definitions(xmr-stak-backend PUBLIC CONF_NO_CUDA)
    message(STATUS "BACKENDS: Disable CUDA")
endif()

###############################################################################
# Define backend: amd
###############################################################################
if(OpenCL_ENABLE)
    # Find OpenCL
    # help systems with a software module system
    list(APPEND CMAKE_PREFIX_PATH "$ENV{AMDAPPSDKROOT}")

    # try to find AMD OpenCL before NVIDIA OpenCL
    find_path(OpenCL_INCLUDE_DIR
        NAMES
            CL/cl.h
            OpenCL/cl.h
        NO_DEFAULT_PATH
        PATHS
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
        NO_DEFAULT_PATH
        PATHS
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
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        ###############################################################################
        # Define target: xmrstak_opencl_backend; OpenCL (AMD) backend shared lib
        ###############################################################################
        add_library(xmrstak_opencl_backend
            SHARED
            "xmrstak/backend/amd/jconf.cpp"
            "xmrstak/backend/amd/amd_gpu/gpu.cpp"
            "xmrstak/backend/amd/minethd.cpp"
        )
        if(XMR-STAK_COMPILE STREQUAL "generic")
            message(STATUS "Forcing OpenCL 1.2 due to XMR-STAK_COMPILE=generic")
            target_compile_definitions(xmrstak_opencl_backend PRIVATE "CONF_ENFORCE_OpenCL_1_2=1")
            set(OpenCL_VERSION_MAJOR "1")
            set(OpenCL_VERSION_MINOR "2")
        endif()
        set(CL_TARGET_OPENCL_VERSION "${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0")
        #target_compile_definitions(xmrstak_opencl_backend PRIVATE "CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION}")
        target_compile_definitions(xmr-stak-backend PUBLIC "CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION}")
        target_link_libraries(xmrstak_opencl_backend xmr-stak-backend "${OpenCL_LIBRARY}")
        #target_link_libraries(xmr-stak-backend xmrstak_opencl_backend)
        include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
        list(APPEND BACKEND_TYPES "amd")
        message(STATUS "BACKENDS: Enable OpenCL"
            " v${OpenCL_VERSION_MAJOR}.${OpenCL_VERSION_MINOR}"
            " @ ${OpenCL_LIBRARY} (${OpenCL_INCLUDE_DIRS})"
        )
    else()
        message(FATAL_ERROR "OpenCL NOT found: use `-DOpenCL_ENABLE=OFF` to build without OpenCL support for AMD gpu's")
    endif()
else()
    target_compile_definitions(xmr-stak-backend PUBLIC CONF_NO_OPENCL)
    message(STATUS "BACKENDS: Disable OpenCL")
endif()
message("* BACKENDS configuration complete!")


################################################################################
# Versioning
################################################################################
message(STATUS "Collecting version information...")

# default known unknowns
set(XMR_STAK_NAME "xmr-stak")
set(XMR_STAK_VERSION "0.0.0")
set(GIT_BRANCH "unknown")
set(GIT_COMMIT_HASH "0000000")

# generate backend string
list(SORT BACKEND_TYPES)
string(REPLACE ";" "-" STR_BACKEND_TYPES "${BACKEND_TYPES}")
target_compile_definitions(xmr-stak-c PUBLIC "BACKEND_TYPE=${STR_BACKEND_TYPES}")

# Find Git and ask about CMAKE_SOURCE_DIR
find_package(Git QUIET)
if(GIT_FOUND)
    # Get the current working branch
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GIT_BRANCH STREQUAL "")
        target_compile_definitions(xmr-stak-c PUBLIC "GIT_BRANCH=${GIT_BRANCH}")
    endif()

    # Get the latest abbreviated commit hash of the working branch
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GIT_COMMIT_HASH STREQUAL "")
        target_compile_definitions(xmr-stak-c PUBLIC "GIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
    endif()
endif()
get_target_property(DEFS xmr-stak-c COMPILE_DEFINITIONS)
string(REGEX REPLACE ";" " -D" DEFS "-D${DEFS}")
set(CMAKE_BINTMP_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp")
file(WRITE "${CMAKE_BINTMP_DIR}/xmr-stak-ver.cpp"
    "#include \"xmrstak/donate-level.hpp\"\n"
    "#include \"xmrstak/version.hpp\"\n"
    "#include <iostream>\n"
    "int main(int argc,char *argv[]){std::cout<< get_version_str() << std::endl;return 0;}"
)
try_compile(BUILD_OK
    ${CMAKE_BINARY_DIR}
    SOURCES "${CMAKE_SOURCE_DIR}/xmrstak/version.cpp" "${CMAKE_BINTMP_DIR}/xmr-stak-ver.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_SOURCE_DIR}"
    COMPILE_DEFINITIONS ${DEFS}
    OUTPUT_VARIABLE BLDLOG
    COPY_FILE ${CMAKE_BINTMP_DIR}/xmr-stak-ver
)
if(BUILD_OK)
    execute_process(
        COMMAND ${CMAKE_BINTMP_DIR}/xmr-stak-ver
        WORKING_DIRECTORY ${CMAKE_BINTMP_DIR}
        OUTPUT_VARIABLE VERSION_LONG
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    message(FATAL_ERROR "${BLDLOG}")
endif()
message(STATUS "Version: ${VERSION_LONG}")
string(REPLACE "/" ";" lst_ver "${VERSION_LONG}")
list(APPEND lst_ver "${PROJECT_NAME}" "0.0.0")
list(GET lst_ver 1 nVersion)
project(xmr-stak VERSION ${nVersion})
message(STATUS "Libs:"
    " [threads:${THREADS_FOUND}]"
    " [hwloc:${HWLOC_FOUND}]"
    " [ssl:${OPENSSL_FOUND}]"
    " [httpd:${MICROHTTPD_FOUND}]"
)
message(STATUS "Backend: [cpu:${CPU_ENABLE}] [amd:${OpenCL_ENABLE}] [nvidia:${CUDA_ENABLE}]")

################################################################################
# Compile & Link
################################################################################

include_directories(BEFORE .)
target_link_libraries(xmr-stak xmr-stak-backend)

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

# activate static libgcc and libstdc++ linking
if(CMAKE_LINK_STATIC)
    set(BUILD_SHARED_LIBRARIES OFF)
    set(DL_LIB ${CMAKE_DL_LIBS})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
        set(LIBS "-static-libgcc -static-libstdc++ ${LIBS}")
    endif()
endif()

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
# vim: et sw=4 sts=4 ts=4:
