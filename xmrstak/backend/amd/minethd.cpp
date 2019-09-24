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
#include "amd_gpu/gpu.hpp"
#include "autoAdjust.hpp"

#include "xmrstak/backend/cpu/crypto/cryptonight.h"
#include "xmrstak/backend/cpu/crypto/cryptonight_aesni.h"
#include "xmrstak/backend/cpu/hwlocMemory.hpp"
#include "xmrstak/backend/cpu/minethd.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/misc/environment.hpp"
#include "xmrstak/misc/executor.hpp"
#include "xmrstak/params.hpp"

#include <assert.h>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace xmrstak
{
namespace amd
{

minethd::minethd(miner_work& pWork, size_t iNo, GpuContext* ctx, const jconf::thd_cfg cfg)
{
	this->backendType = iBackend::AMD;
	oWork = pWork;
	bQuit = 0;
	iThreadNo = (uint8_t)iNo;
	this->iGpuIndex = cfg.index;
	iJobNo = 0;
	iHashCount = 0;
	iTimestamp = 0;
	pGpuCtx = ctx;
	this->affinity = cfg.cpu_aff;
	autoTune = jconf::inst()->GetAutoTune();

	std::unique_lock<std::mutex> lck(thd_aff_set);
	std::future<void> order_guard = order_fix.get_future();

	oWorkThd = std::thread(&minethd::work_main, this);

	order_guard.wait();

	if(affinity >= 0) //-1 means no affinity
		if(!cpu::minethd::thd_setaffinity(oWorkThd.native_handle(), affinity))
			printer::inst()->print_msg(L1, "WARNING setting affinity failed.");
}

extern "C"
{
#ifdef WIN32
	__declspec(dllexport)
#endif
		std::vector<iBackend*>* xmrstak_start_backend(uint32_t threadOffset, miner_work& pWork, environment& env)
	{
		environment::inst(&env);
		return amd::minethd::thread_starter(threadOffset, pWork);
	}
} // extern "C"

bool minethd::init_gpus()
{
	size_t i, n = jconf::inst()->GetThreadCount();

	printer::inst()->print_msg(L1, "Compiling code and initializing GPUs. This will take a while...");
	vGpuData.resize(n);

	jconf::thd_cfg cfg;
	for(i = 0; i < n; i++)
	{
		jconf::inst()->GetThreadConfig(i, cfg);
		vGpuData[i].deviceIdx = cfg.index;
		vGpuData[i].rawIntensity = cfg.intensity;
		vGpuData[i].workSize = cfg.w_size;
		vGpuData[i].stridedIndex = cfg.stridedIndex;
		vGpuData[i].memChunk = cfg.memChunk;
		vGpuData[i].compMode = cfg.compMode;
		vGpuData[i].unroll = cfg.unroll;
		vGpuData[i].interleave = cfg.interleave;
	}

	return InitOpenCL(vGpuData.data(), n, jconf::inst()->GetPlatformIdx()) == ERR_SUCCESS;
}

std::vector<GpuContext> minethd::vGpuData;

std::vector<iBackend*>* minethd::thread_starter(uint32_t threadOffset, miner_work& pWork)
{
	std::vector<iBackend*>* pvThreads = new std::vector<iBackend*>();

	if(!configEditor::file_exist(params::inst().configFileAMD))
	{
		autoAdjust adjust;
		if(!adjust.printConfig())
			return pvThreads;
	}

	if(!jconf::inst()->parse_config())
	{
		win_exit();
	}

	// \ todo get device count and exit if no opencl device

	if(!init_gpus())
	{
		printer::inst()->print_msg(L1, "WARNING: AMD device not found");
		return pvThreads;
	}

	size_t i, n = jconf::inst()->GetThreadCount();
	pvThreads->reserve(n);

	jconf::thd_cfg cfg;
	for(i = 0; i < n; i++)
	{
		jconf::inst()->GetThreadConfig(i, cfg);

		const std::string backendName = xmrstak::params::inst().openCLVendor;

		if(cfg.cpu_aff >= 0)
		{
#if defined(__APPLE__)
			printer::inst()->print_msg(L1, "WARNING on macOS thread affinity is only advisory.");
#endif

			printer::inst()->print_msg(L1, "Starting %s GPU (OpenCL) thread %d, affinity: %d.", backendName.c_str(), i, (int)cfg.cpu_aff);
		}
		else
			printer::inst()->print_msg(L1, "Starting %s GPU (OpenCL) thread %d, no affinity.", backendName.c_str(), i);

		minethd* thd = new minethd(pWork, i + threadOffset, &vGpuData[i], cfg);
		pvThreads->push_back(thd);
	}

	return pvThreads;
}

void minethd::work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.unlock();
	std::this_thread::yield();

	cryptonight_ctx* cpu_ctx;
	cpu_ctx = cpu::minethd::minethd_alloc_ctx();

	if(cpu_ctx == nullptr)
	{
		printer::inst()->print_msg(L0, "ERROR: miner was not able to allocate memory, miner will be stopped.");
		win_exit(1);
	}
	// start with root algorithm and switch later if fork version is reached
	auto miner_algo = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot();

	cpu::minethd::cn_on_new_job set_job;

	cpu::minethd::func_multi_selector<1>(&cpu_ctx, set_job, ::jconf::inst()->HaveHardwareAes(), true /*bNoPrefetch*/, miner_algo);

	uint8_t version = 0;
	size_t lastPoolId = 0;

	pGpuCtx->maxRawIntensity = pGpuCtx->rawIntensity;

	if(autoTune != 0)
	{
		pGpuCtx->rawIntensity = pGpuCtx->computeUnits * pGpuCtx->workSize;
		pGpuCtx->rawIntensity = std::min(pGpuCtx->maxRawIntensity, pGpuCtx->rawIntensity);
	}
	// parameters needed for auto tuning
	uint32_t cntTestRounds = 0;
	uint64_t accRuntime = 0;
	double bestHashrate = 0.0;
	uint32_t bestIntensity = pGpuCtx->maxRawIntensity;

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

		size_t round_ctr = 0;

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));
		uint64_t target = oWork.iTarget;

		XMRSetJob(pGpuCtx, oWork.bWorkBlob, oWork.iWorkSize, target, miner_algo, cpu_ctx->cn_r_ctx.height);

		if(oWork.bNiceHash)
			pGpuCtx->Nonce = *(uint32_t*)(oWork.bWorkBlob + 39);

		while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			//Allocate a new nonce every 16 rounds
			if((round_ctr++ & 0xF) == 0)
			{
				globalStates::inst().calc_start_nonce(pGpuCtx->Nonce, oWork.bNiceHash, pGpuCtx->rawIntensity * 16);
				// check if the job is still valid, there is a small possibility that the job is switched
				if(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) != iJobNo)
					break;
			}

			// if auto tuning is running we will not adjust the interleave interval
			const bool adjustInterleave = autoTune == 0;
			uint64_t t0 = interleaveAdjustDelay(pGpuCtx, adjustInterleave);

			cl_uint results[0x100];
			memset(results, 0, sizeof(cl_uint) * (0x100));

			XMRRunJob(pGpuCtx, results, miner_algo);

			for(size_t i = 0; i < results[0xFF]; i++)
			{
				uint8_t bWorkBlob[128];
				uint8_t bResult[32];

				memcpy(bWorkBlob, oWork.bWorkBlob, oWork.iWorkSize);
				memset(bResult, 0, sizeof(job_result::bResult));

				*(uint32_t*)(bWorkBlob + 39) = results[i];

				cpu_ctx->hash_fn(bWorkBlob, oWork.iWorkSize, bResult, &cpu_ctx, miner_algo);
				if((*((uint64_t*)(bResult + 24))) < oWork.iTarget)
					executor::inst()->push_event(ex_event(job_result(oWork.sJobID, results[i], bResult, iThreadNo, miner_algo), oWork.iPoolId));
				else
					executor::inst()->push_event(ex_event("AMD Invalid Result", pGpuCtx->deviceIdx, oWork.iPoolId));
			}

			updateStats(pGpuCtx->rawIntensity, oWork.iPoolId);

			accRuntime += updateTimings(pGpuCtx, t0);

			// tune intensity
			if(autoTune != 0)
			{
				if(cntTestRounds++ == autoTune)
				{
					double avgHashrate = static_cast<double>(cntTestRounds * pGpuCtx->rawIntensity) / (static_cast<double>(accRuntime) / 1000.0);
					if(avgHashrate > bestHashrate)
					{
						bestHashrate = avgHashrate;
						bestIntensity = pGpuCtx->rawIntensity;
					}

					// increase always in workSize steps to avoid problems with the compatibility mode
					pGpuCtx->rawIntensity += pGpuCtx->workSize;
					// trigger that we query for new nonce's because the number of nonce previous allocated depends on the rawIntensity
					round_ctr = 0x10;

					if(pGpuCtx->rawIntensity > pGpuCtx->maxRawIntensity)
					{
						// lock intensity to the best values
						autoTune = 0;
						pGpuCtx->rawIntensity = bestIntensity;
						printer::inst()->print_msg(L1, "OpenCL %u|%u: lock intensity at %u",
							pGpuCtx->deviceIdx,
							pGpuCtx->idWorkerOnDevice,
							bestIntensity);
					}
					else
					{
						printer::inst()->print_msg(L1, "OpenCL %u|%u: auto-tune validate intensity %u|%u",
							pGpuCtx->deviceIdx,
							pGpuCtx->idWorkerOnDevice,
							pGpuCtx->rawIntensity,
							bestIntensity);
					}
					// update gpu with new intensity
					XMRSetJob(pGpuCtx, oWork.bWorkBlob, oWork.iWorkSize, target, miner_algo, cpu_ctx->cn_r_ctx.height);
				}
				// use 3 rounds to warm up with the new intensity
				else if(cntTestRounds == autoTune + 3)
				{
					// reset values for the next test period
					cntTestRounds = 0;
					accRuntime = 0;
				}
			}

			std::this_thread::yield();
		}

		globalStates::inst().consume_work(oWork, iJobNo);
	}
}

} // namespace amd
} // namespace xmrstak
