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
	typedef void (*cn_hash_fun_multi)(const void*, size_t, void*, cryptonight_ctx**);
	static cn_hash_fun_multi func_multi_selector(size_t N, bool bHaveAes, bool bNoPrefetch, bool mineMonero);

	minethd(miner_work& pWork, size_t iNo, int iMultiway, bool no_prefetch, int64_t affinity);

	void multiway_work_main(size_t N, cn_hash_fun_multi hash_fun_multi);

	void work_main();
	void double_work_main();
	void triple_work_main();
	void quad_work_main();
	void penta_work_main();

	void consume_work();
	void prep_multiway_work(size_t N, uint8_t *bWorkBlob, uint32_t **piNonce);

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
