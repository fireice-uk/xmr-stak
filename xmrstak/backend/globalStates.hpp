#pragma once

#include "miner_work.hpp"
#include "xmrstak/misc/environment.hpp"

#include <atomic>


namespace xmrstak
{

struct globalStates
{

	static inline globalStates& inst()
	{
		auto& env = environment::inst();
		if(env.pglobalStates == nullptr)
			env.pglobalStates = new globalStates;
		return *env.pglobalStates;
	}

	void switch_work(miner_work& pWork);

	miner_work oGlobalWork;
	std::atomic<uint64_t> iGlobalJobNo;
	std::atomic<uint64_t> iConsumeCnt;
	uint64_t iThreadCount;

	private:

	globalStates() : iThreadCount(0)
	{
	}
	
};

} // namepsace xmrstak
