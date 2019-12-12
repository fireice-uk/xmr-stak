#pragma once

#include <stdint.h>
#include <string>

#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"

#include <cuda.h>

typedef struct
{
	int device_id;
	const char* device_name;
	int device_arch[2];
	int device_mpcount;
	int device_blocks;
	int device_threads;
	int device_bfactor;
	int device_bsleep;
	int device_maxThreadsPerBlock;
	int syncMode;

	uint32_t* d_input;
	uint32_t inputlen;
	uint32_t* d_result_count;
	uint32_t* d_result_nonce;
	uint32_t* d_long_state;
	uint32_t* d_ctx_state;

	std::string name;
	size_t free_device_memory;
	size_t total_device_memory;

	//randomx stuff
	uint8_t rx_dataset_seedhash[32] = {0};
	uint32_t *d_rx_dataset = nullptr;
	uint32_t *d_rx_hashes = nullptr;
	uint32_t *d_rx_entropy = nullptr;
	uint32_t *d_rx_vm_states = nullptr;
	uint32_t *d_rx_rounding = nullptr;
	size_t d_scratchpads_size = 0u;

} nvid_ctx;

extern "C"
{

	/** get device count
 *
 * @param deviceCount[out] cuda device count
 * @return error code: 0 == error is occurred, 1 == no error
 */
	int cuda_get_devicecount(int* deviceCount);
	int cuda_get_deviceinfo(nvid_ctx* ctx);
	int cryptonight_extra_cpu_init(nvid_ctx* ctx);
	void cryptonight_extra_cpu_set_data(nvid_ctx* ctx, const void* data, uint32_t len);

}

void randomx_prepare(nvid_ctx *ctx, const uint8_t* seed_hash, const xmrstak_algo& miner_algo, uint32_t batch_size);

namespace RandomX_Monero  { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Wownero { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Loki    { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Arqma   { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }

