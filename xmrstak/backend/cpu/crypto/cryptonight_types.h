#ifndef __CRYPTONIGHT_TYPES_H_INCLUDED
#define __CRYPTONIGHT_TYPES_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <cstdint>

typedef struct {
	uint8_t hash_state[224]; // Need only 200, explicit align
	uint8_t* long_state;
	uint8_t ctx_info[24]; //Use some of the extra memory for flags
} cryptonight_ctx;

typedef struct {
	const char* warning;
} alloc_msg;

#ifdef __cplusplus
}
#endif

#endif
