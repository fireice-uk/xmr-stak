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
#include "amd_gpu/gpu.hpp"

#include "xmrstak/backend/cpu/crypto/cryptonight_aesni.h"
#include "xmrstak/backend/cpu/crypto/cryptonight.h"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/backend/cpu/minethd.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/executor.hpp"
#include "xmrstak/misc/environment.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/backend/cpu/hwlocMemory.hpp"

#include <assert.h>
#include <cmath>
#include <chrono>
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
	iJobNo = 0;
	iHashCount = 0;
	iTimestamp = 0;
	pGpuCtx = ctx;
	this->affinity = cfg.cpu_aff;

	std::unique_lock<std::mutex> lck(thd_aff_set);
	std::future<void> order_guard = order_fix.get_future();

	oWorkThd = std::thread(&minethd::work_main, this);

	order_guard.wait();

	if(affinity >= 0) //-1 means no affinity
		if(!cpu::minethd::thd_setaffinity(oWorkThd.native_handle(), affinity))
			printer::inst()->print_msg(L1, "WARNING setting affinity failed.");
}

extern "C"  {
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
	for (i = 0; i < n; i++)
	{
		jconf::inst()->GetThreadConfig(i, cfg);
		
		if(cfg.cpu_aff >= 0)
		{
#if defined(__APPLE__)
			printer::inst()->print_msg(L1, "WARNING on MacOS thread affinity is only advisory.");
#endif

			printer::inst()->print_msg(L1, "Starting AMD GPU thread %d, affinity: %d.", i, (int)cfg.cpu_aff);
		}
		else
			printer::inst()->print_msg(L1, "Starting AMD GPU thread %d, no affinity.", i);

		minethd* thd = new minethd(pWork, i + threadOffset, &vGpuData[i], cfg);
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
	globalStates::inst().iConsumeCnt++;

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

		uint32_t h_per_round = pGpuCtx->rawIntensity;
		size_t round_ctr = 0;

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));
		uint64_t target = oWork.iTarget;
		XMRSetJob(pGpuCtx, oWork.bWorkBlob, oWork.iWorkSize, target);

		if(oWork.bNiceHash)
			pGpuCtx->Nonce = *(uint32_t*)(oWork.bWorkBlob + 39);

		while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			//Allocate a new nonce every 16 rounds
			if((round_ctr++ & 0xF) == 0)
			{
				globalStates::inst().calc_start_nonce(pGpuCtx->Nonce, oWork.bNiceHash, h_per_round * 16);
			}

			cl_uint results[0x100];
			memset(results,0,sizeof(cl_uint)*(0x100));

			XMRRunJob(pGpuCtx, results);

			for(size_t i = 0; i < results[0xFF]; i++)
			{
				uint8_t	bWorkBlob[112];
				uint8_t	bResult[32];

				memcpy(bWorkBlob, oWork.bWorkBlob, oWork.iWorkSize);
				memset(bResult, 0, sizeof(job_result::bResult));

				*(uint32_t*)(bWorkBlob + 39) = results[i];

				hash_fun(bWorkBlob, oWork.iWorkSize, bResult, cpu_ctx);
				if ( (*((uint64_t*)(bResult + 24))) < oWork.iTarget)
					executor::inst()->push_event(ex_event(job_result(oWork.sJobID, results[i], bResult, iThreadNo), oWork.iPoolId));
				else
					executor::inst()->push_event(ex_event("AMD Invalid Result", oWork.iPoolId));
			}

			iCount += pGpuCtx->rawIntensity;
			uint64_t iStamp = get_timestamp_ms();
			iHashCount.store(iCount, std::memory_order_relaxed);
			iTimestamp.store(iStamp, std::memory_order_relaxed);
			std::this_thread::yield();
		}

		consume_work();
	}
}

} // namespace amd
} // namespace xmrstak
