
list(APPEND HEADERS_CRYPTO
#    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/rx/Rx.h
#    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/rx/RxAlgo.h
#    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/rx/RxCache.h
#    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/rx/RxConfig.h
#    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/rx/RxDataset.h
#    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/rx/RxVm.h
)

list(APPEND SOURCES_RANDOMX
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/aes_hash.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/allocator.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/argon2_core.c
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/argon2_ref.c
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/blake2_generator.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/blake2/blake2b.c
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/bytecode_machine.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/dataset.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/instructions_portable.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/randomx.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/reciprocal.c
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/soft_aes.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/superscalar.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/virtual_machine.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/virtual_memory.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/vm_compiled_light.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/vm_compiled.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/vm_interpreted_light.cpp
    ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/vm_interpreted.cpp
)

if(CMAKE_C_COMPILER_ID MATCHES MSVC)
    enable_language(ASM_MASM)
    list(APPEND SOURCES_RANDOMX
         ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/jit_compiler_x86_static.asm
         ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/jit_compiler_x86.cpp
         ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/common/VirtualMemory_win.cpp
        )
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    list(APPEND SOURCES_RANDOMX
         ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/jit_compiler_x86_static.S
         ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/jit_compiler_x86.cpp
         ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/common/VirtualMemory_unix.cpp
        )
    # cheat because cmake and ccache hate each other
    set_property(SOURCE ${CMAKE_SOURCE_DIR}/xmrstak/backend/cpu/crypto/randomx/jit_compiler_x86_static.S PROPERTY LANGUAGE C)
endif()

add_library(xmr-stak-randomx
    STATIC
    ${SOURCES_RANDOMX}
)
set_property(TARGET xmr-stak-randomx PROPERTY POSITION_INDEPENDENT_CODE ON)