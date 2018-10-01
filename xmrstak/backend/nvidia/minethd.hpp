#pragma once

#include "xmrstak/jconf.hpp"
#include "jconf.hpp"
#include "nvcc_code/cryptonight.hpp"

#include "xmrstak/backend/cpu/minethd.hpp"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/misc/environment.hpp"

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <future>


namespace xmrstak
{
namespace nvidia
{

class minethd : public iBackend
{
public:

	static std::vector<iBackend*>* thread_starter(uint32_t threadOffset, miner_work& pWork);
	static bool self_test();

private:
	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx**);

	minethd(miner_work& pWork, size_t iNo, const jconf::thd_cfg& cfg);
	void start_mining();

	void work_main();

	static std::atomic<uint64_t> iGlobalJobNo;
	static std::atomic<uint64_t> iConsumeCnt;
	static uint64_t iThreadCount;
	uint64_t iJobNo;

	miner_work oWork;

	std::promise<void> numa_promise;
	std::promise<void> thread_work_promise;

	// block thread until all NVIDIA GPUs are initialized
	std::future<void> thread_work_guard;

	std::thread oWorkThd;
	int64_t affinity;

	nvid_ctx ctx;

	bool bQuit;
};

} // namespace nvidia
} // namespace xmrstak
