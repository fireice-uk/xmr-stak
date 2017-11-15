#pragma once

#include "crypto/cryptonight.h"
#include "xmrstak/backend/miner_work.hpp"
#include "xmrstak/backend/iBackend.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <future>

namespace xmrstak
{
namespace cpu
{

class minethd : public iBackend
{
public:
	static std::vector<iBackend*> thread_starter(uint32_t threadOffset, miner_work& pWork);
	static bool self_test();

	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx*);

	static cn_hash_fun func_selector(bool bHaveAes, bool bNoPrefetch, bool mineMonero);
	static bool thd_setaffinity(std::thread::native_handle_type h, uint64_t cpu_id);

	static cryptonight_ctx* minethd_alloc_ctx();

private:

	typedef void (*cn_hash_fun_dbl)(const void*, size_t, void*, cryptonight_ctx* __restrict, cryptonight_ctx* __restrict);
	static cn_hash_fun_dbl func_dbl_selector(bool bHaveAes, bool bNoPrefetch, bool mineMonero);

	minethd(miner_work& pWork, size_t iNo, bool double_work, bool no_prefetch, int64_t affinity);

	void work_main();
	void double_work_main();
	void consume_work();
	uint32_t* prep_double_work(uint8_t bDoubleWorkBlob[sizeof(miner_work::bWorkBlob) * 2]);

	uint64_t iJobNo;

	static miner_work oGlobalWork;
	miner_work oWork;

	std::promise<void> order_fix;
	std::mutex thd_aff_set;

	std::thread oWorkThd;
	int64_t affinity;

	bool bQuit;
	bool bNoPrefetch;
};

} // namespace cpu
} // namepsace xmrstak
