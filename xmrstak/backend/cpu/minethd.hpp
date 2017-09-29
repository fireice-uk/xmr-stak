#pragma once

#include "crypto/cryptonight.h"
#include "xmrstak/backend/miner_work.hpp"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/backend/globalStates.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

namespace xmrstak
{
namespace cpu
{

class minethd : public IBackend
{
public:
	static std::vector<IBackend*> thread_starter(uint32_t threadOffset, miner_work& pWork);
	static bool self_test();

	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx*);

	static cn_hash_fun func_selector(bool bHaveAes, bool bNoPrefetch);
	static void thd_setaffinity(std::thread::native_handle_type h, uint64_t cpu_id);

	static cryptonight_ctx* minethd_alloc_ctx();

private:

	typedef void (*cn_hash_fun_dbl)(const void*, size_t, void*, cryptonight_ctx* __restrict, cryptonight_ctx* __restrict);
	static cn_hash_fun_dbl func_dbl_selector(bool bHaveAes, bool bNoPrefetch);

	minethd(miner_work& pWork, size_t iNo, bool double_work, bool no_prefetch, int64_t affinity);

	// We use the top 10 bits of the nonce for thread and resume
	// This allows us to resume up to 128 threads 4 times before
	// we get nonce collisions
	// Bottom 22 bits allow for an hour of work at 1000 H/s
	inline uint32_t calc_start_nonce(uint32_t resume)
	{
		return reverseBits<uint32_t>(static_cast<uint32_t>(iThreadNo + GlobalStates::inst().iThreadCount * resume));
	}

	// Limited version of the nonce calc above
	inline uint32_t calc_nicehash_nonce(uint32_t start, uint32_t resume)
	{ 
		return start | ( ( reverseBits<uint32_t>(static_cast<uint32_t>(iThreadNo + GlobalStates::inst().iThreadCount * resume)) >> 4u ) );
	}

	void work_main();
	void double_work_main();
	void consume_work();

	uint64_t iJobNo;

	static miner_work oGlobalWork;
	miner_work oWork;

	void pin_thd_affinity();
	// Held by the creating context to prevent a race cond with oWorkThd = std::thread(...)
	std::mutex work_thd_mtx;

	std::thread oWorkThd;
	uint8_t iThreadNo;
	int64_t affinity;

	bool bQuit;
	bool bNoPrefetch;
};

} // namespace cpu
} // namepsace xmrstak
