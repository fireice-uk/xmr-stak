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
#include "xmrstak/backend/cpu/crypto/cryptonight.h"
#include "xmrstak/backend/cpu/crypto/cryptonight_aesni.h"
#include "xmrstak/backend/cpu/hwlocMemory.hpp"
#include "xmrstak/backend/cpu/minethd.hpp"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/misc/environment.hpp"
#include "xmrstak/misc/executor.hpp"
#include "xmrstak/misc/utility.hpp"
#include "xmrstak/params.hpp"

#include <assert.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#ifndef USE_PRECOMPILED_HEADERS
#ifdef WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/types.h>
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
void* lib_handle;
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
	ctx.memMode = cfg.memMode;
	this->affinity = cfg.cpu_aff;

	std::future<void> numa_guard = numa_promise.get_future();
	thread_work_guard = thread_work_promise.get_future();

	oWorkThd = std::thread(&minethd::work_main, this);

	/* Wait until the gpu memory is initialized and numa cpu memory is pinned.
	 * The startup time is reduced if the memory is initialized in sequential order
	 * without concurrent threads (CUDA driver is less occupied).
	 */
	numa_guard.wait();
}

void minethd::start_mining()
{
	thread_work_promise.set_value();
	if(this->affinity >= 0) //-1 means no affinity
		if(!cpu::minethd::thd_setaffinity(oWorkThd.native_handle(), affinity))
			printer::inst()->print_msg(L1, "WARNING setting affinity failed.");
}

bool minethd::self_test()
{
	return true;
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

	auto miner_algo = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot();

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
		std::cout << "WARNING: NVIDIA no device found" << std::endl;
		return pvThreads;
	}
	else
	{
		std::cout << "NVIDIA: found " << deviceCount << " potential device's" << std::endl;
	}

	size_t i, n = jconf::inst()->GetGPUThreadCount();
	pvThreads->reserve(n);

	cuInit(0);

	jconf::thd_cfg cfg;
	for(i = 0; i < n; i++)
	{
		jconf::inst()->GetGPUThreadConfig(i, cfg);

		if(cfg.cpu_aff >= 0)
		{
#if defined(__APPLE__)
			printer::inst()->print_msg(L1, "WARNING on macOS thread affinity is only advisory.");
#endif

			printer::inst()->print_msg(L1, "Starting NVIDIA GPU thread %d, affinity: %d.", i, (int)cfg.cpu_aff);
		}
		else
			printer::inst()->print_msg(L1, "Starting NVIDIA GPU thread %d, no affinity.", i);

		minethd* thd = new minethd(pWork, i + threadOffset, cfg);
		pvThreads->push_back(thd);
	}

	for(i = 0; i < n; i++)
	{
		static_cast<minethd*>((*pvThreads)[i])->start_mining();
	}

	return pvThreads;
}

void minethd::work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	if(cuda_get_deviceinfo(&ctx) != 0 || cryptonight_extra_cpu_init(&ctx) != 1)
	{
		printer::inst()->print_msg(L0, "Setup failed for GPU %d. Exiting.\n", (int)iThreadNo);
		std::exit(0);
	}

	// numa memory bind and gpu memory is initialized
	numa_promise.set_value();

	std::this_thread::yield();
	// wait until all NVIDIA devices are initialized
	thread_work_guard.wait();

	cryptonight_ctx* cpu_ctx;
	cpu_ctx = cpu::minethd::minethd_alloc_ctx();

	// start with root algorithm and switch later if fork version is reached
	auto miner_algo = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot();

	cpu::minethd::cn_on_new_job set_job;
	cpu::minethd::func_multi_selector<1>(&cpu_ctx, set_job, ::jconf::inst()->HaveHardwareAes(), true /*bNoPrefetch*/, miner_algo);

	uint32_t iNonce;

	uint8_t version = 0;
	size_t lastPoolId = 0;

	while(bQuit == 0)
	{
		if(oWork.bStall)
		{
			/* We are stalled here because the executor didn't find a job for us yet,
			 * either because of network latency, or a socket problem. Since we are
			 * raison d'etre of this software it us sensible to just wait until we have something
			 */

			while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
				std::this_thread::sleep_for(std::chrono::milliseconds(100));

			globalStates::inst().consume_work(oWork, iJobNo);
			continue;
		}
		uint8_t new_version = oWork.getVersion();
		if(new_version != version || oWork.iPoolId != lastPoolId)
		{
			coinDescription coinDesc = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(oWork.iPoolId);
			if(new_version >= coinDesc.GetMiningForkVersion())
			{
				miner_algo = coinDesc.GetMiningAlgo();
				cpu::minethd::func_multi_selector<1>(&cpu_ctx, set_job, ::jconf::inst()->HaveHardwareAes(), true /*bNoPrefetch*/, miner_algo);
			}
			else
			{
				miner_algo = coinDesc.GetMiningAlgoRoot();
				cpu::minethd::func_multi_selector<1>(&cpu_ctx, set_job, ::jconf::inst()->HaveHardwareAes(), true /*bNoPrefetch*/, miner_algo);
			}
			lastPoolId = oWork.iPoolId;
			version = new_version;
		}

		if(set_job != nullptr)
			set_job(oWork, &cpu_ctx);

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
				// check if the job is still valid, there is a small possibility that the job is switched
				if(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) != iJobNo)
					break;
			}

			uint32_t foundNonce[10];
			uint32_t foundCount;

			cryptonight_extra_cpu_prepare(&ctx, iNonce, miner_algo);

			cryptonight_core_cpu_hash(&ctx, miner_algo, iNonce, cpu_ctx->cn_r_ctx.height);

			cryptonight_extra_cpu_final(&ctx, iNonce, oWork.iTarget, &foundCount, foundNonce, miner_algo);

			for(size_t i = 0; i < foundCount; i++)
			{

				uint8_t bWorkBlob[128];
				uint8_t bResult[32];

				memcpy(bWorkBlob, oWork.bWorkBlob, oWork.iWorkSize);
				memset(bResult, 0, sizeof(job_result::bResult));

				*(uint32_t*)(bWorkBlob + 39) = foundNonce[i];

				cpu_ctx->hash_fn(bWorkBlob, oWork.iWorkSize, bResult, &cpu_ctx, miner_algo);
				if((*((uint64_t*)(bResult + 24))) < oWork.iTarget)
					executor::inst()->push_event(ex_event(job_result(oWork.sJobID, foundNonce[i], bResult, iThreadNo, miner_algo), oWork.iPoolId));
				else
					executor::inst()->push_event(ex_event("NVIDIA Invalid Result", ctx.device_id, oWork.iPoolId));
			}

			iNonce += h_per_round;
			updateStats(h_per_round, oWork.iPoolId);
			std::this_thread::yield();
		}

		globalStates::inst().consume_work(oWork, iJobNo);
	}
}

} // namespace nvidia
} // namespace xmrstak
