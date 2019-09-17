#pragma once
#include <inttypes.h>
#include <stddef.h>

#include "xmrstak/misc/environment.hpp"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/backend/cpu/crypto/randomx/randomx.h"
#include "xmrstak/misc/console.hpp"
#include <mutex>
#include <array>
#include <atomic>
#include <thread>

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

struct cryptonight_ctx
{
	uint8_t hash_state[224]; // Need only 200, explicit align
	uint8_t* long_state;
	uint8_t ctx_info[24]; //Use some of the extra memory for flags
	cn_mainloop_fun loop_fn = nullptr;
	cn_hash_fun hash_fn = nullptr;
	uint8_t* fun_data = nullptr;
	xmrstak_algo last_algo = invalid_algo;

	randomx_vm* m_rx_vm = nullptr;
};


struct randomX_global_ctx
{
	static randomX_global_ctx & inst()
	{
		auto& env = xmrstak::environment::inst();
		if(env.pGlobalCtx == nullptr)
		{
			std::unique_lock<std::mutex> lck(env.update);
			if(env.pGlobalCtx == nullptr)
				env.pGlobalCtx = new randomX_global_ctx;
		}
		return *env.pGlobalCtx;
	}

	randomx_dataset* getDataset()
	{
		return m_rx_dataset;
	}

	void updateDataset(const std::array<uint8_t, 32>& seed_hash, const uint32_t num_threads)
	{
		// Check if we need to update cache and dataset
		if(m_rx_seed_hash == seed_hash)
			return;

		const uint32_t thread_id = m_rx_dataset_init_thread_counter++;
		printer::inst()->print_msg(LDEBUG,"Thread %u started updating RandomX dataset %x", thread_id,&m_rx_dataset_init_thread_counter);

		// Wait for all threads to get here
		do
		{
			std::this_thread::yield();
		} while (m_rx_dataset_init_thread_counter.load() != num_threads);

		// One of the threads updates cache
		{
			std::lock_guard<std::mutex> g(m_rx_dataset_lock);
			if(m_rx_seed_hash != seed_hash)
			{
				m_rx_seed_hash = seed_hash;
				randomx_init_cache(m_rx_cache, m_rx_seed_hash.data(), m_rx_seed_hash.size());
			}
		}

		// All threads update dataset
		const uint32_t a = (randomx_dataset_item_count() * static_cast<uint64_t>(thread_id)) / num_threads;
		const uint32_t b = (randomx_dataset_item_count() * (static_cast<uint64_t>(thread_id) + 1u)) / num_threads;
		printer::inst()->print_msg(LDEBUG,"Thread %u start updating RandomX dataset %u %u", thread_id, a, b);
		randomx_init_dataset(m_rx_dataset, m_rx_cache, a, b - a);

		printer::inst()->print_msg(LDEBUG,"Thread %u finished updating RandomX dataset", thread_id);

		// Wait for all threads to complete
		--m_rx_dataset_init_thread_counter;
		do {
		    std::this_thread::yield();
		} while (m_rx_dataset_init_thread_counter.load() != 0);
	}

private:
	randomX_global_ctx() : m_rx_dataset_init_thread_counter(0u)
	{
		randomx_dataset* dataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);
		if (!dataset) {
			dataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
		}
		m_rx_cache = randomx_alloc_cache(static_cast<randomx_flags>(RANDOMX_FLAG_JIT | RANDOMX_FLAG_LARGE_PAGES));
		if (!m_rx_cache) {
			m_rx_cache = randomx_alloc_cache(RANDOMX_FLAG_JIT);
		}
		m_rx_dataset = dataset;
	}

	std::mutex m_rx_dataset_lock;
	randomx_cache* m_rx_cache = nullptr;
	randomx_dataset* m_rx_dataset = nullptr;
	std::array<uint8_t, 32> m_rx_seed_hash = {{0}};
	std::atomic<uint32_t> m_rx_dataset_init_thread_counter;
};

struct alloc_msg
{
	const char* warning;
};

size_t cryptonight_init(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
cryptonight_ctx* cryptonight_alloc_ctx(size_t use_fast_mem, size_t use_mlock, alloc_msg* msg);
void cryptonight_free_ctx(cryptonight_ctx* ctx);
