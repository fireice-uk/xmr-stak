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

	typedef void (*cn_on_new_job)(const miner_work&, cryptonight_ctx**);

	static void func_selector(cryptonight_ctx**, bool bHaveAes, bool bNoPrefetch, const xmrstak_algo& algo);
	static bool thd_setaffinity(std::thread::native_handle_type h, uint64_t cpu_id);
	static bool thd_setlowpriority(std::thread::native_handle_type h);

	static cryptonight_ctx* minethd_alloc_ctx();

	template<size_t N>
	static void func_multi_selector(cryptonight_ctx**, minethd::cn_on_new_job& on_new_job,
			bool bHaveAes, bool bNoPrefetch, const xmrstak_algo& algo, const std::string& asm_version_str = "off");

	private:
		
	minethd(miner_work& pWork, size_t iNo,size_t iOffset, int iMultiway, bool no_prefetch, int64_t affinity, const std::string& asm_version);

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

	int64_t affinity;

	bool bNoPrefetch;
	std::string asm_version_str = "off";
};

} // namespace cpu
} // namespace xmrstak
