#pragma once
#include <atomic>
#include "miner_work.h"

namespace xmrstak
{

struct GlobalStates
{

	static void switch_work(miner_work& pWork);

	static miner_work oGlobalWork;
	static std::atomic<uint64_t> iGlobalJobNo;
	static std::atomic<uint64_t> iConsumeCnt;
	static uint64_t iThreadCount;
	
};

} // namepsace xmrstak
