#pragma once

#include "xmrstak/jconf.hpp"
#include "jconf.hpp"
#include "nvcc_code/cryptonight.h"

#include "xmrstak/bakcend/cpu/crypto/cryptonight.h"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/misc/environment.hpp"

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>


namespace xmrstak
{
namespace nvidia
{

class minethd : public IBackend
{
public:

	static void switch_work(miner_work& pWork);
	static std::vector<IBackend*>* thread_starter(uint32_t threadOffset, miner_work& pWork);
	static bool self_test();

private:
	typedef void (*cn_hash_fun)(const void*, size_t, void*, cryptonight_ctx*);
	
	minethd(miner_work& pWork, size_t iNo, const jconf::thd_cfg& cfg);

	// We use the top 10 bits of the nonce for thread and resume
	// This allows us to resume up to 128 threads 4 times before
	// we get nonce collisions
	// Bottom 22 bits allow for an hour of work at 1000 H/s
	inline uint32_t calc_start_nonce(uint32_t resume)
	{
		return reverseBits<uint32_t>(iThreadNo + GlobalStates::inst().iThreadCount * resume);
	}

	// Limited version of the nonce calc above
	inline uint32_t calc_nicehash_nonce(uint32_t start, uint32_t resume)
	{
		return start | ( ( reverseBits(iThreadNo + GlobalStates::inst().iThreadCount * resume) >> 4u ) );
	}

	void work_main();
	void consume_work();

	static std::atomic<uint64_t> iGlobalJobNo;
	static std::atomic<uint64_t> iConsumeCnt;
	static uint64_t iThreadCount;
	uint64_t iJobNo;

	static miner_work oGlobalWork;
	miner_work oWork;

	std::thread oWorkThd;
	uint8_t iThreadNo;

	nvid_ctx ctx;

	bool bQuit;
};

} // namespace nvidia
} // namepsace xmrstak
