#pragma once

#include "amd_gpu/gpu.hpp"
#include "jconf.hpp"
#include "xmrstak/backend/cpu/crypto/cryptonight.h"
#include "xmrstak/backend/miner_work.hpp"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/misc/environment.hpp"

#include <thread>
#include <atomic>
#include <future>

namespace xmrstak
{
namespace amd
{

class minethd  : public iBackend
{
public:

	static std::vector<iBackend*>* thread_starter(uint32_t threadOffset, miner_work& pWork);
	static bool init_gpus();

private:
	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx**);

	minethd(miner_work& pWork, size_t iNo, GpuContext* ctx, const jconf::thd_cfg cfg);

	void work_main();

	uint64_t iJobNo;

	miner_work oWork;

	std::promise<void> order_fix;
	std::mutex thd_aff_set;

	std::thread oWorkThd;
	int64_t affinity;

	bool bQuit;
	bool bNoPrefetch;

	//Mutable ptr to vector below, different for each thread
	GpuContext* pGpuCtx;

	// WARNING - this vector (but not its contents) must be immutable
	// once the threads are started
	static std::vector<GpuContext> vGpuData;
};

} // namespace amd
} // namespace xmrstak
