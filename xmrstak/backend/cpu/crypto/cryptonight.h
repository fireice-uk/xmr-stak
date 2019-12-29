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
#include <winsock2.h>
#include <windows.h>
#include <ntsecapi.h>
#else
#define ABI_ATTRIBUTE __attribute__((ms_abi))
#endif

#include "xmrstak/backend/cpu/hwlocHelper.hpp"
#include "cryptonight_1.h"

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
	uint32_t numa = 0;

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

	randomx_dataset* getDataset(size_t numaId)
	{
		return m_rx_datasets[numaId];
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
			std::lock_guard<std::mutex> g(m_rx_cache_lock);
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
		size_t numElements = b - a;
		randomx_init_dataset(m_rx_datasets[0], m_rx_cache, a, numElements);
		for(size_t i = 1; i < m_rx_datasets.size(); ++i)
		{
			if(m_rx_datasets[i] != nullptr)
			{
				memcpy((uint8_t*)getRandomXDataset(i) + a * 64u, (uint8_t*)getRandomXDataset(0) + a * 64u, numElements * 64u);
			}
		}

		printer::inst()->print_msg(LDEBUG,"Thread %u finished updating RandomX dataset", thread_id);

		// Wait for all threads to complete
		--m_rx_dataset_init_thread_counter;
		do {
		    std::this_thread::yield();
		} while (m_rx_dataset_init_thread_counter.load() != 0);
	}

	void init(size_t numaId)
	{
		{
			std::unique_lock<std::mutex> lck(dataset_locks[numaId]);
			if(m_rx_datasets[numaId])
			{
				printer::inst()->print_msg(LDEBUG,"dataset/cache already created for numa %u", uint32_t(numaId));
				return;
			}
			printer::inst()->print_msg(LDEBUG,"allocate dataset/cache for numa %u", uint32_t(numaId));
	#ifdef __linux__
			randomx_dataset* dataset = randomx_alloc_dataset(static_cast<randomx_flags>(RANDOMX_FLAG_LARGE_PAGES | RANDOMX_FLAG_LARGE_PAGES_1G));
			if (!dataset)
			{
				printer::inst()->print_msg(LDEBUG,"Warning: dataset allocation with 1 GiB pages failed");
	#else
				randomx_dataset* dataset = nullptr;
	#endif
				dataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);
				if (!dataset)
				{
					printer::inst()->print_msg(LDEBUG,"Warning: dataset allocation with 2 MiB pages failed");
					dataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
					printer::inst()->print_msg(LDEBUG,"dataset allocated without huge pages");
				}
				else
					printer::inst()->print_msg(LDEBUG,"dataset allocated with 2 MiB pages");
	#ifdef __linux__
			}
			else
				printer::inst()->print_msg(LDEBUG,"dataset allocated with 1 GiB pages");
	#endif

			m_rx_datasets[numaId] = dataset;
		}
		{
			std::unique_lock<std::mutex> lck(m_rx_cache_lock);
			if(numaId == 0 && m_rx_cache == nullptr)
			{
				m_rx_cache = randomx_alloc_cache(static_cast<randomx_flags>(RANDOMX_FLAG_JIT | RANDOMX_FLAG_LARGE_PAGES));
				if (!m_rx_cache) {
					m_rx_cache = randomx_alloc_cache(RANDOMX_FLAG_JIT);
				}
			}
		}
	}

	void release(size_t numaId)
	{
		{
			std::unique_lock<std::mutex> lck(dataset_locks[numaId]);
			if(!m_rx_datasets[numaId])
			{
				printer::inst()->print_msg(LDEBUG,"dataset/cache for numa %u alreday released", uint32_t(numaId));
				return;
			}
			printer::inst()->print_msg(LDEBUG,"release dataset/cache for numa %u", uint32_t(numaId));
			randomx_release_dataset(m_rx_datasets[numaId]);
			m_rx_datasets[numaId] = nullptr;
		}
		{
			std::unique_lock<std::mutex> lck(m_rx_cache_lock);
			if(numaId == 0 && m_rx_cache)
			{
				randomx_release_cache(m_rx_cache);
				m_rx_cache = nullptr;
			}
		}
	}

private:
	randomX_global_ctx() : m_rx_dataset_init_thread_counter(0u)
	{
		size_t numNumaNodes = getNumNumaNodes();
		m_rx_datasets.resize(numNumaNodes, nullptr);
		dataset_locks.reset(new std::mutex[numNumaNodes]);
	}

	std::mutex m_rx_cache_lock;
	randomx_cache* m_rx_cache = nullptr;
	std::unique_ptr<std::mutex[]> dataset_locks;
	std::vector<randomx_dataset*> m_rx_datasets;
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
