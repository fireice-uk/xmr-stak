#pragma once

#include "miner_work.hpp"
#include "xmrstak/misc/environment.hpp"
#include "xmrstak/misc/console.hpp"

#include <atomic>


namespace xmrstak
{

struct pool_data
{
	uint32_t iSavedNonce;
	size_t   pool_id;
 
	pool_data() : iSavedNonce(0), pool_id(0)
	{
	}
};

struct globalStates
{
	static inline globalStates& inst()
	{
		auto& env = environment::inst();
		if(env.pglobalStates == nullptr)
			env.pglobalStates = new globalStates;
		return *env.pglobalStates;
	}

	//pool_data is in-out winapi style
	void switch_work(miner_work& pWork, pool_data& dat);

	inline uint32_t calc_start_nonce(uint32_t nicehash_nonce, uint32_t reserve_count)
	{
		uint32_t debug_nonce = nicehash_nonce | iGlobalNonce.fetch_add(reserve_count);
		printer::inst()->print_msg(L1, "DEBUG: start_nonce assigned nh: %.8x rc: %.8x nonce: %.8x", nicehash_nonce, reserve_count, debug_nonce);
	}

	miner_work oGlobalWork;
	std::atomic<uint64_t> iGlobalJobNo;
	std::atomic<uint64_t> iConsumeCnt;
	std::atomic<uint32_t> iGlobalNonce;
	uint64_t iThreadCount;
	size_t pool_id;

private:
	globalStates() : iThreadCount(0)
	{
	}
};

} // namepsace xmrstak
