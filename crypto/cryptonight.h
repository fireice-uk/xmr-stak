#ifndef __CRYPTONIGHT_H_INCLUDED
#define __CRYPTONIGHT_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <inttypes.h>

#define MEMORY  2097152

typedef struct {
	uint8_t hash_state[224]; // Need only 200, explicit align
	uint8_t* long_state;
	uint8_t ctx_info[24]; //Use some of the extra memory for flags
} cryptonight_ctx;

typedef struct {
	const char* warning;
} alloc_msg;

size_t cryptonight_init(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
cryptonight_ctx* cryptonight_alloc_ctx(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
void cryptonight_free_ctx(cryptonight_ctx* ctx);

void cryptonight_hash_ctx(const void* input, size_t len, void* output, cryptonight_ctx* ctx);
void cryptonight_hash_ctx_np(const void* input, size_t len, void* output, cryptonight_ctx* ctx);
void cryptonight_double_hash_ctx(const void*  input, size_t len, void* output, cryptonight_ctx* __restrict ctx0, cryptonight_ctx* __restrict ctx1);

#ifdef __cplusplus
}
#endif

#endif
