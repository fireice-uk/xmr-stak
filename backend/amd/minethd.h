#pragma once
#include <thread>
#include <atomic>
#include "./jconf.h"
#include "../IBackend.hpp"

#include "amd_gpu/gpu.h"

namespace xmrstak
{
namespace amd
{

class minethd  : public IBackend
{
public:

	static void switch_work(miner_work& pWork);
	static std::vector<IBackend*>* thread_starter(uint32_t threadOffset, miner_work& pWork);
	static bool init_gpus();

private:
	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx*);
	
	minethd(miner_work& pWork, size_t iNo, GpuContext* ctx);

	// We use the top 8 bits of the nonce for thread and resume
	// This allows us to resume up to 64 threads 4 times before
	// we get nonce collisions
	// Bottom 24 bits allow for an hour of work at 4000 H/s
	inline uint32_t calc_start_nonce(uint32_t resume)
	{
		return reverseBits<uint32_t>(static_cast<uint32_t>(iThreadNo + GlobalStates::iThreadCount * resume));
	}
	
	void work_main();
	void double_work_main();
	void consume_work();

	uint64_t iJobNo;

	static miner_work oGlobalWork;
	miner_work oWork;

	std::thread oWorkThd;
	uint8_t iThreadNo;

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
