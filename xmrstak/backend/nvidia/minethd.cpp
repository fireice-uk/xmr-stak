/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
  *
  */

#include "minethd.hpp"
#include "autoAdjust.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/backend/cpu/crypto/cryptonight_aesni.h"
#include "xmrstak/backend/cpu/crypto/cryptonight.h"
#include "xmrstak/backend/cpu/minethd.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/misc/executor.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/environment.hpp"
#include "xmrstak/backend/cpu/hwlocMemory.hpp"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/misc/utility.hpp"

#include <assert.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <bitset>
#include <vector>

#ifndef USE_PRECOMPILED_HEADERS
#ifdef WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <dlfcn.h>
#endif
#include <iostream>
#endif

namespace xmrstak
{
namespace nvidia
{

#ifdef WIN32
	HINSTANCE lib_handle;
#else
	void *lib_handle;
#endif

minethd::minethd(miner_work& pWork, size_t iNo, const jconf::thd_cfg& cfg)
{
	this->backendType = iBackend::NVIDIA;
	oWork = pWork;
	bQuit = 0;
	iThreadNo = (uint8_t)iNo;
	iJobNo = 0;

	ctx.device_id = (int)cfg.id;
	ctx.device_blocks = (int)cfg.blocks;
	ctx.device_threads = (int)cfg.threads;
	ctx.device_bfactor = (int)cfg.bfactor;
	ctx.device_bsleep = (int)cfg.bsleep;
	ctx.syncMode = cfg.syncMode;
	this->affinity = cfg.cpu_aff;

	std::unique_lock<std::mutex> lck(thd_aff_set);
	std::future<void> order_guard = order_fix.get_future();

	oWorkThd = std::thread(&minethd::work_main, this);

	order_guard.wait();

	if(affinity >= 0) //-1 means no affinity
		if(!cpu::minethd::thd_setaffinity(oWorkThd.native_handle(), affinity))
			printer::inst()->print_msg(L1, "WARNING setting affinity failed.");
}


bool minethd::self_test()
{
	cryptonight_ctx* ctx0;
	unsigned char out[32];
	bool bResult = true;

	ctx0 = new cryptonight_ctx;
	if(::jconf::inst()->HaveHardwareAes())
	{
		//cryptonight_hash_ctx("This is a test", 14, out, ctx0);
		bResult = memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;
	}
	else
	{
		//cryptonight_hash_ctx_soft("This is a test", 14, out, ctx0);
		bResult = memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;
	}
	delete ctx0;

	//if(!bResult)
	//	printer::inst()->print_msg(L0,
	//	"Cryptonight hash self-test failed. This might be caused by bad compiler optimizations.");

	return bResult;
}


extern "C"
{
#ifdef WIN32
__declspec(dllexport)
#endif
std::vector<iBackend*>* xmrstak_start_backend(uint32_t threadOffset, miner_work& pWork, environment& env)
{
	environment::inst(&env);
	return nvidia::minethd::thread_starter(threadOffset, pWork);
}
} // extern "C"

std::vector<iBackend*>* minethd::thread_starter(uint32_t threadOffset, miner_work& pWork)
{
	std::vector<iBackend*>* pvThreads = new std::vector<iBackend*>();

	if(!configEditor::file_exist(params::inst().configFileNVIDIA))
	{
		autoAdjust adjust;
		if(!adjust.printConfig())
			return pvThreads;
	}

	if(!jconf::inst()->parse_config())
	{
		win_exit();
	}

	int deviceCount = 0;
	if(cuda_get_devicecount(&deviceCount) != 1)
	{
		std::cout<<"WARNING: NVIDIA no device found"<<std::endl;
		return pvThreads;
	}

	size_t i, n = jconf::inst()->GetGPUThreadCount();
	pvThreads->reserve(n);

	jconf::thd_cfg cfg;
	for (i = 0; i < n; i++)
	{
		jconf::inst()->GetGPUThreadConfig(i, cfg);

		if(cfg.cpu_aff >= 0)
		{
#if defined(__APPLE__)
			printer::inst()->print_msg(L1, "WARNING on MacOS thread affinity is only advisory.");
#endif

			printer::inst()->print_msg(L1, "Starting NVIDIA GPU thread %d, affinity: %d.", i, (int)cfg.cpu_aff);
		}
		else
			printer::inst()->print_msg(L1, "Starting NVIDIA GPU thread %d, no affinity.", i);
		
		minethd* thd = new minethd(pWork, i + threadOffset, cfg);
		pvThreads->push_back(thd);

	}

	return pvThreads;
}

void minethd::switch_work(miner_work& pWork)
{
	// iConsumeCnt is a basic lock-like polling mechanism just in case we happen to push work
	// faster than threads can consume them. This should never happen in real life.
	// Pool cant physically send jobs faster than every 250ms or so due to net latency.

	while (globalStates::inst().iConsumeCnt.load(std::memory_order_seq_cst) < globalStates::inst().iThreadCount)
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	globalStates::inst().oGlobalWork = pWork;
	globalStates::inst().iConsumeCnt.store(0, std::memory_order_seq_cst);
	globalStates::inst().iGlobalJobNo++;
}

void minethd::consume_work()
{
	memcpy(&oWork, &globalStates::inst().oGlobalWork, sizeof(miner_work));
	iJobNo++;
	globalStates::inst().iConsumeCnt++;
}

void minethd::work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.release();
	std::this_thread::yield();

	uint64_t iCount = 0;
	cryptonight_ctx* cpu_ctx;
	cpu_ctx = cpu::minethd::minethd_alloc_ctx();
	cn_hash_fun hash_fun = cpu::minethd::func_selector(::jconf::inst()->HaveHardwareAes(), true /*bNoPrefetch*/, ::jconf::inst()->IsCurrencyMonero());
	uint32_t iNonce;

	globalStates::inst().iConsumeCnt++;

	if(cuda_get_deviceinfo(&ctx) != 0 || cryptonight_extra_cpu_init(&ctx) != 1)
	{
		printer::inst()->print_msg(L0, "Setup failed for GPU %d. Exitting.\n", (int)iThreadNo);
		std::exit(0);
	}

	bool mineMonero = strcmp_i(::jconf::inst()->GetCurrency(), "monero");

	while (bQuit == 0)
	{
		if (oWork.bStall)
		{
			/* We are stalled here because the executor didn't find a job for us yet,
			 * either because of network latency, or a socket problem. Since we are
			 * raison d'etre of this software it us sensible to just wait until we have something
			 */

			while (globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
				std::this_thread::sleep_for(std::chrono::milliseconds(100));

			consume_work();
			continue;
		}

		cryptonight_extra_cpu_set_data(&ctx, oWork.bWorkBlob, oWork.iWorkSize);

		uint32_t h_per_round = ctx.device_blocks * ctx.device_threads;
		size_t round_ctr = 0;

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));

		if(oWork.bNiceHash)
			iNonce = *(uint32_t*)(oWork.bWorkBlob + 39);

		while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			//Allocate a new nonce every 16 rounds
			if((round_ctr++ & 0xF) == 0)
			{
				globalStates::inst().calc_start_nonce(iNonce, oWork.bNiceHash, h_per_round * 16);
			}
			
			uint32_t foundNonce[10];
			uint32_t foundCount;

			cryptonight_extra_cpu_prepare(&ctx, iNonce);

			cryptonight_core_cpu_hash(&ctx, mineMonero);

			cryptonight_extra_cpu_final(&ctx, iNonce, oWork.iTarget, &foundCount, foundNonce);

			for(size_t i = 0; i < foundCount; i++)
			{

				uint8_t	bWorkBlob[112];
				uint8_t	bResult[32];

				memcpy(bWorkBlob, oWork.bWorkBlob, oWork.iWorkSize);
				memset(bResult, 0, sizeof(job_result::bResult));

				*(uint32_t*)(bWorkBlob + 39) = foundNonce[i];

				hash_fun(bWorkBlob, oWork.iWorkSize, bResult, cpu_ctx);
				if ( (*((uint64_t*)(bResult + 24))) < oWork.iTarget)
					executor::inst()->push_event(ex_event(job_result(oWork.sJobID, foundNonce[i], bResult, iThreadNo), oWork.iPoolId));
				else
					executor::inst()->push_event(ex_event("NVIDIA Invalid Result", oWork.iPoolId));
			}

			iCount += h_per_round;
			iNonce += h_per_round;

			using namespace std::chrono;
			uint64_t iStamp = get_timestamp_ms();
			iHashCount.store(iCount, std::memory_order_relaxed);
			iTimestamp.store(iStamp, std::memory_order_relaxed);
			std::this_thread::yield();
		}

		consume_work();
	}
}

} // namespace xmrstak

} //namespace nvidia
