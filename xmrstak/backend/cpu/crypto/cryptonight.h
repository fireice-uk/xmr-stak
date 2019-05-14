#pragma once
#include <inttypes.h>
#include <stddef.h>

#include "variant4_random_math.h"

#if defined _MSC_VER
#define ABI_ATTRIBUTE
#else
#define ABI_ATTRIBUTE __attribute__((ms_abi))
#endif

struct cryptonight_ctx;

typedef void (*cn_mainloop_fun)(cryptonight_ctx* ctx);
typedef void (*cn_double_mainloop_fun)(cryptonight_ctx*, cryptonight_ctx*);
typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx**, const xmrstak_algo&);

void v4_compile_code(size_t N, cryptonight_ctx* ctx, int code_size);

struct extra_ctx_r
{
	uint64_t height = 0;
	// the buffer must be able to hold NUM_INSTRUCTIONS_MAX and a termination instruction
	V4_Instruction code[NUM_INSTRUCTIONS_MAX + 1];
};

struct cryptonight_ctx
{
	uint8_t hash_state[224]; // Need only 200, explicit align
	uint8_t* long_state;
	uint8_t ctx_info[24]; //Use some of the extra memory for flags
	cn_mainloop_fun loop_fn = nullptr;
	cn_hash_fun hash_fn = nullptr;
	uint8_t* fun_data = nullptr;
	int asm_version = 0;
	xmrstak_algo last_algo = invalid_algo;

	union {
		extra_ctx_r cn_r_ctx;
	};
};

struct alloc_msg
{
	const char* warning;
};

size_t cryptonight_init(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
cryptonight_ctx* cryptonight_alloc_ctx(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
void cryptonight_free_ctx(cryptonight_ctx* ctx);
