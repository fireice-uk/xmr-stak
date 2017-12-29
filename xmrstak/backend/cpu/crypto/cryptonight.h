#ifndef __CRYPTONIGHT_H_INCLUDED
#define __CRYPTONIGHT_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include "cryptonight_types.h"
#include <stddef.h>
#include <inttypes.h>
#include "xmrstak/backend/cryptonight.hpp"

size_t cryptonight_init(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
cryptonight_ctx* cryptonight_alloc_ctx(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
void cryptonight_free_ctx(cryptonight_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
