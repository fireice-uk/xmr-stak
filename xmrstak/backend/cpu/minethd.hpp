#pragma once

#include "xmrstak/jconf.hpp"
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

	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx**);

	static cn_hash_fun func_selector(bool bHaveAes, bool bNoPrefetch, xmrstak_algo algo);
	static bool thd_setaffinity(std::thread::native_handle_type h, uint64_t cpu_id);

	static cryptonight_ctx* minethd_alloc_ctx();

private:

	template<size_t N>
	static cn_hash_fun func_multi_selector(bool bHaveAes, bool bNoPrefetch, xmrstak_algo algo, const std::string& asm_version_str = "off");

	minethd(miner_work& pWork, size_t iNo, int iMultiway, bool no_prefetch, int64_t affinity, const std::string& asm_version);

	template<uint32_t N>
	void multiway_work_main();

	template<size_t N>
	void prep_multiway_work(uint8_t *bWorkBlob, uint32_t **piNonce);

	void work_main();
	void double_work_main();
	void triple_work_main();
	void quad_work_main();
	void penta_work_main();

	uint64_t iJobNo;

	miner_work oWork;

	std::promise<void> order_fix;
	std::mutex thd_aff_set;

	std::thread oWorkThd;
	int64_t affinity;

	bool bQuit;
	bool bNoPrefetch;
	std::string asm_version_str = "off";
};

} // namespace cpu
} // namespace xmrstak
