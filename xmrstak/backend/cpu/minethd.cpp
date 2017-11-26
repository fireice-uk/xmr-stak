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

#include "crypto/cryptonight_aesni.h"

#include "xmrstak/misc/console.hpp"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/backend//globalStates.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/params.hpp"
#include "jconf.hpp"

#include "xmrstak/misc/executor.hpp"
#include "minethd.hpp"
#include "xmrstak/jconf.hpp"

#include "hwlocMemory.hpp"
#include "xmrstak/backend/miner_work.hpp"

#ifndef CONF_NO_HWLOC
#   include "autoAdjustHwloc.hpp"
#else
#   include "autoAdjust.hpp"
#endif

#include <assert.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <thread>
#include <bitset>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>

#if defined(__APPLE__)
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#define SYSCTL_CORE_COUNT   "machdep.cpu.core_count"
#elif defined(__FreeBSD__)
#include <pthread_np.h>
#endif //__APPLE__

#endif //_WIN32

namespace xmrstak
{
namespace cpu
{

bool minethd::thd_setaffinity(std::thread::native_handle_type h, uint64_t cpu_id)
{
#if defined(_WIN32)
	return SetThreadAffinityMask(h, 1ULL << cpu_id) != 0;
#elif defined(__APPLE__)
	thread_port_t mach_thread;
	thread_affinity_policy_data_t policy = { static_cast<integer_t>(cpu_id) };
	mach_thread = pthread_mach_thread_np(h);
	return thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1) == KERN_SUCCESS;
#elif defined(__FreeBSD__)
	cpuset_t mn;
	CPU_ZERO(&mn);
	CPU_SET(cpu_id, &mn);
	return pthread_setaffinity_np(h, sizeof(cpuset_t), &mn) == 0;
#else
	cpu_set_t mn;
	CPU_ZERO(&mn);
	CPU_SET(cpu_id, &mn);
	return pthread_setaffinity_np(h, sizeof(cpu_set_t), &mn) == 0;
#endif
}

minethd::minethd(miner_work& pWork, size_t iNo, bool double_work, bool no_prefetch, int64_t affinity)
{
	this->backendType = iBackend::CPU;
	oWork = pWork;
	bQuit = 0;
	iThreadNo = (uint8_t)iNo;
	iJobNo = 0;
	bNoPrefetch = no_prefetch;
	this->affinity = affinity;

	std::unique_lock<std::mutex> lck(thd_aff_set);
	std::future<void> order_guard = order_fix.get_future();

	if(double_work)
		oWorkThd = std::thread(&minethd::double_work_main, this);
	else
		oWorkThd = std::thread(&minethd::work_main, this);

	order_guard.wait();

	if(affinity >= 0) //-1 means no affinity
		if(!thd_setaffinity(oWorkThd.native_handle(), affinity))
			printer::inst()->print_msg(L1, "WARNING setting affinity failed.");
}

cryptonight_ctx* minethd::minethd_alloc_ctx()
{
	cryptonight_ctx* ctx;
	alloc_msg msg = { 0 };

	switch (::jconf::inst()->GetSlowMemSetting())
	{
	case ::jconf::never_use:
		ctx = cryptonight_alloc_ctx(1, 1, &msg);
		if (ctx == NULL)
			printer::inst()->print_msg(L0, "MEMORY ALLOC FAILED: %s", msg.warning);
		return ctx;

	case ::jconf::no_mlck:
		ctx = cryptonight_alloc_ctx(1, 0, &msg);
		if (ctx == NULL)
			printer::inst()->print_msg(L0, "MEMORY ALLOC FAILED: %s", msg.warning);
		return ctx;

	case ::jconf::print_warning:
		ctx = cryptonight_alloc_ctx(1, 1, &msg);
		if (msg.warning != NULL)
			printer::inst()->print_msg(L0, "MEMORY ALLOC FAILED: %s", msg.warning);
		if (ctx == NULL)
			ctx = cryptonight_alloc_ctx(0, 0, NULL);
		return ctx;

	case ::jconf::always_use:
		return cryptonight_alloc_ctx(0, 0, NULL);

	case ::jconf::unknown_value:
		return NULL; //Shut up compiler
	}

	return nullptr; //Should never happen
}

bool minethd::self_test()
{
	alloc_msg msg = { 0 };
	size_t res;
	bool fatal = false;

	switch (::jconf::inst()->GetSlowMemSetting())
	{
	case ::jconf::never_use:
		res = cryptonight_init(1, 1, &msg);
		fatal = true;
		break;

	case ::jconf::no_mlck:
		res = cryptonight_init(1, 0, &msg);
		fatal = true;
		break;

	case ::jconf::print_warning:
		res = cryptonight_init(1, 1, &msg);
		break;

	case ::jconf::always_use:
		res = cryptonight_init(0, 0, &msg);
		break;

	case ::jconf::unknown_value:
	default:
		return false; //Shut up compiler
	}

	if(msg.warning != nullptr)
		printer::inst()->print_msg(L0, "MEMORY INIT ERROR: %s", msg.warning);

	if(res == 0 && fatal)
		return false;

	cryptonight_ctx *ctx0, *ctx1;
	if((ctx0 = minethd_alloc_ctx()) == nullptr)
		return false;

	if((ctx1 = minethd_alloc_ctx()) == nullptr)
	{
		cryptonight_free_ctx(ctx0);
		return false;
	}

	bool bResult = true;

	bool mineMonero = ::jconf::inst()->IsCurrencyMonero();
	if(mineMonero)
	{
		unsigned char out[64];
		cn_hash_fun hashf;
		cn_hash_fun_dbl hashdf;


		hashf = func_selector(::jconf::inst()->HaveHardwareAes(), false, mineMonero);
		hashf("This is a test", 14, out, ctx0);
		bResult = memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;

		hashf = func_selector(::jconf::inst()->HaveHardwareAes(), true, mineMonero);
		hashf("This is a test", 14, out, ctx0);
		bResult &= memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;

		hashdf = func_dbl_selector(::jconf::inst()->HaveHardwareAes(), false, mineMonero);
		hashdf("The quick brown fox jumps over the lazy dogThe quick brown fox jumps over the lazy log", 43, out, ctx0, ctx1);
		bResult &= memcmp(out, "\x3e\xbb\x7f\x9f\x7d\x27\x3d\x7c\x31\x8d\x86\x94\x77\x55\x0c\xc8\x00\xcf\xb1\x1b\x0c\xad\xb7\xff\xbd\xf6\xf8\x9f\x3a\x47\x1c\x59"
							   "\xb4\x77\xd5\x02\xe4\xd8\x48\x7f\x42\xdf\xe3\x8e\xed\x73\x81\x7a\xda\x91\xb7\xe2\x63\xd2\x91\x71\xb6\x5c\x44\x3a\x01\x2a\x41\x22", 64) == 0;

		hashdf = func_dbl_selector(::jconf::inst()->HaveHardwareAes(), true, mineMonero);
		hashdf("The quick brown fox jumps over the lazy dogThe quick brown fox jumps over the lazy log", 43, out, ctx0, ctx1);
		bResult &= memcmp(out, "\x3e\xbb\x7f\x9f\x7d\x27\x3d\x7c\x31\x8d\x86\x94\x77\x55\x0c\xc8\x00\xcf\xb1\x1b\x0c\xad\xb7\xff\xbd\xf6\xf8\x9f\x3a\x47\x1c\x59"
							   "\xb4\x77\xd5\x02\xe4\xd8\x48\x7f\x42\xdf\xe3\x8e\xed\x73\x81\x7a\xda\x91\xb7\xe2\x63\xd2\x91\x71\xb6\x5c\x44\x3a\x01\x2a\x41\x22", 64) == 0;
	}
	cryptonight_free_ctx(ctx0);
	cryptonight_free_ctx(ctx1);

	if(!bResult)
		printer::inst()->print_msg(L0,
			"Cryptonight hash self-test failed. This might be caused by bad compiler optimizations.");

	return bResult;
}

std::vector<iBackend*> minethd::thread_starter(uint32_t threadOffset, miner_work& pWork)
{
	std::vector<iBackend*> pvThreads;

	if(!configEditor::file_exist(params::inst().configFileCPU))
	{
		autoAdjust adjust;
		if(!adjust.printConfig())
			return pvThreads;
	}

	if(!jconf::inst()->parse_config())
	{
		win_exit();
	}


	//Launch the requested number of single and double threads, to distribute
	//load evenly we need to alternate single and double threads
	size_t i, n = jconf::inst()->GetThreadCount();
	pvThreads.reserve(n);

	jconf::thd_cfg cfg;
	for (i = 0; i < n; i++)
	{
		jconf::inst()->GetThreadConfig(i, cfg);

		if(cfg.iCpuAff >= 0)
		{
#if defined(__APPLE__)
			printer::inst()->print_msg(L1, "WARNING on MacOS thread affinity is only advisory.");
#endif

			printer::inst()->print_msg(L1, "Starting %s thread, affinity: %d.", cfg.bDoubleMode ? "double" : "single", (int)cfg.iCpuAff);
		}
		else
			printer::inst()->print_msg(L1, "Starting %s thread, no affinity.", cfg.bDoubleMode ? "double" : "single");
		
		minethd* thd = new minethd(pWork, i + threadOffset, cfg.bDoubleMode, cfg.bNoPrefetch, cfg.iCpuAff);
		pvThreads.push_back(thd);
	}

	return pvThreads;
}

void minethd::consume_work()
{
	memcpy(&oWork, &globalStates::inst().inst().oGlobalWork, sizeof(miner_work));
	iJobNo++;
	globalStates::inst().inst().iConsumeCnt++;
}

minethd::cn_hash_fun minethd::func_selector(bool bHaveAes, bool bNoPrefetch, bool mineMonero)
{
	// We have two independent flag bits in the functions
	// therefore we will build a binary digit and select the
	// function as a two digit binary
	// Digit order SOFT_AES, NO_PREFETCH, MINER_ALGO

	static const cn_hash_fun func_table[] = {
		/* there will be 8 function entries if `CONF_NO_MONERO` and `CONF_NO_AEON`
		 * is not defined. If one is defined there will be 4 entries.
		 */
#ifndef CONF_NO_MONERO
		cryptonight_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, false, false>,
		cryptonight_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, false, true>,
		cryptonight_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, true, false>,
		cryptonight_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, true, true>
#endif
#if (!defined(CONF_NO_AEON)) && (!defined(CONF_NO_MONERO))
		// comma will be added only if Monero and Aeon is build
		,
#endif
#ifndef CONF_NO_AEON
		cryptonight_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, false, false>,
		cryptonight_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, false, true>,
		cryptonight_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, true, false>,
		cryptonight_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, true, true>
#endif
	};

	std::bitset<3> digit;
	digit.set(0, !bNoPrefetch);
	digit.set(1, !bHaveAes);

	// define aeon settings
#if defined(CONF_NO_AEON) || defined(CONF_NO_MONERO)
	// ignore 3rd bit if only on currency is active
	digit.set(2, 0);
#else
	digit.set(2, !mineMonero);
#endif

	return func_table[digit.to_ulong()];
}

void minethd::work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.release();
	std::this_thread::yield();

	cn_hash_fun hash_fun;
	cryptonight_ctx* ctx;
	uint64_t iCount = 0;
	uint64_t* piHashVal;
	uint32_t* piNonce;
	job_result result;

	hash_fun = func_selector(::jconf::inst()->HaveHardwareAes(), bNoPrefetch, ::jconf::inst()->IsCurrencyMonero());
	ctx = minethd_alloc_ctx();

	piHashVal = (uint64_t*)(result.bResult + 24);
	piNonce = (uint32_t*)(oWork.bWorkBlob + 39);
	globalStates::inst().inst().iConsumeCnt++;
	result.iThreadId = iThreadNo;

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

		size_t nonce_ctr = 0;
		constexpr size_t nonce_chunk = 4096; // Needs to be a power of 2

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));
		memcpy(result.sJobID, oWork.sJobID, sizeof(job_result::sJobID));

		if(oWork.bNiceHash)
			result.iNonce = *piNonce;

		while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			if ((iCount++ & 0xF) == 0) //Store stats every 16 hashes
			{
				using namespace std::chrono;
				uint64_t iStamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
				iHashCount.store(iCount, std::memory_order_relaxed);
				iTimestamp.store(iStamp, std::memory_order_relaxed);
			}

			if((nonce_ctr++ & (nonce_chunk-1)) == 0)
			{
				globalStates::inst().calc_start_nonce(result.iNonce, oWork.bNiceHash, nonce_chunk);
			}

			*piNonce = ++result.iNonce;

			hash_fun(oWork.bWorkBlob, oWork.iWorkSize, result.bResult, ctx);

			if (*piHashVal < oWork.iTarget)
				executor::inst()->push_event(ex_event(result, oWork.iPoolId));

			std::this_thread::yield();
		}

		consume_work();
	}

	cryptonight_free_ctx(ctx);
}

minethd::cn_hash_fun_dbl minethd::func_dbl_selector(bool bHaveAes, bool bNoPrefetch, bool mineMonero)
{
	// We have two independent flag bits in the functions
	// therefore we will build a binary digit and select the
	// function as a two digit binary
	// Digit order SOFT_AES, NO_PREFETCH, MINER_ALGO

	static const cn_hash_fun_dbl func_table[] = {
		/* there will be 8 function entries if `CONF_NO_MONERO` and `CONF_NO_AEON`
		 * is not defined. If one is defined there will be 4 entries.
		 */
#ifndef CONF_NO_MONERO
		cryptonight_double_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, false, false>,
		cryptonight_double_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, false, true>,
		cryptonight_double_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, true, false>,
		cryptonight_double_hash<MONERO_MASK, MONERO_ITER, MONERO_MEMORY, true, true>
#endif
#if (!defined(CONF_NO_AEON)) && (!defined(CONF_NO_MONERO))
		// comma will be added only if Monero and Aeon is build
		,
#endif
#ifndef CONF_NO_AEON
		cryptonight_double_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, false, false>,
		cryptonight_double_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, false, true>,
		cryptonight_double_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, true, false>,
		cryptonight_double_hash<AEON_MASK, AEON_ITER, AEON_MEMORY, true, true>
#endif
	};

	std::bitset<3> digit;
	digit.set(0, !bNoPrefetch);
	digit.set(1, !bHaveAes);

	// define aeon settings
#if defined(CONF_NO_AEON) || defined(CONF_NO_MONERO)
	// ignore 3rd bit if only on currency is active
	digit.set(2, 0);
#else
	digit.set(2, !mineMonero);
#endif

	return func_table[digit.to_ulong()];
}

uint32_t* minethd::prep_double_work(uint8_t bDoubleWorkBlob[sizeof(miner_work::bWorkBlob) * 2])
{
	memcpy(bDoubleWorkBlob, oWork.bWorkBlob, oWork.iWorkSize);
	memcpy(bDoubleWorkBlob + oWork.iWorkSize, oWork.bWorkBlob, oWork.iWorkSize);
	return (uint32_t*)(bDoubleWorkBlob + oWork.iWorkSize + 39);
}

void minethd::double_work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.release();
	std::this_thread::yield();

	cn_hash_fun_dbl hash_fun;
	cryptonight_ctx* ctx0;
	cryptonight_ctx* ctx1;
	uint64_t iCount = 0;
	uint64_t *piHashVal0, *piHashVal1;
	uint32_t *piNonce0, *piNonce1;
	uint8_t bDoubleHashOut[64];
	uint8_t bDoubleWorkBlob[sizeof(miner_work::bWorkBlob) * 2];
	uint32_t iNonce;
	job_result res;

	hash_fun = func_dbl_selector(::jconf::inst()->HaveHardwareAes(), bNoPrefetch, ::jconf::inst()->IsCurrencyMonero());
	ctx0 = minethd_alloc_ctx();
	ctx1 = minethd_alloc_ctx();

	piHashVal0 = (uint64_t*)(bDoubleHashOut + 24);
	piHashVal1 = (uint64_t*)(bDoubleHashOut + 32 + 24);
	piNonce0 = (uint32_t*)(bDoubleWorkBlob + 39);

	if(!oWork.bStall)
		piNonce1 = prep_double_work(bDoubleWorkBlob);
	else
		piNonce1 = nullptr;

	globalStates::inst().inst().iConsumeCnt++;

	while (bQuit == 0)
	{
		if (oWork.bStall)
		{
			/*	We are stalled here because the executor didn't find a job for us yet,
			either because of network latency, or a socket problem. Since we are
			raison d'etre of this software it us sensible to just wait until we have something*/

			while (globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
				std::this_thread::sleep_for(std::chrono::milliseconds(100));

			consume_work();
			piNonce1 = prep_double_work(bDoubleWorkBlob);
			continue;
		}

		size_t nonce_ctr = 0;
		constexpr size_t nonce_chunk = 4096; //Needs to be a power of 2

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));

		if(oWork.bNiceHash)
			iNonce = *piNonce0;

		while (globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			if ((iCount & 0x7) == 0) //Store stats every 16 hashes
			{
				using namespace std::chrono;
				uint64_t iStamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
				iHashCount.store(iCount, std::memory_order_relaxed);
				iTimestamp.store(iStamp, std::memory_order_relaxed);
			}
			iCount += 2;
			
			
			if((nonce_ctr++ & (nonce_chunk/2 - 1)) == 0)
			{
				globalStates::inst().calc_start_nonce(iNonce, oWork.bNiceHash, nonce_chunk);
			}

			*piNonce0 = ++iNonce;
			*piNonce1 = ++iNonce;

			hash_fun(bDoubleWorkBlob, oWork.iWorkSize, bDoubleHashOut, ctx0, ctx1);

			if (*piHashVal0 < oWork.iTarget)
				executor::inst()->push_event(ex_event(job_result(oWork.sJobID, iNonce-1, bDoubleHashOut, iThreadNo), oWork.iPoolId));

			if (*piHashVal1 < oWork.iTarget)
				executor::inst()->push_event(ex_event(job_result(oWork.sJobID, iNonce, bDoubleHashOut + 32, iThreadNo), oWork.iPoolId));

			std::this_thread::yield();
		}

		consume_work();
		piNonce1 = prep_double_work(bDoubleWorkBlob);
	}

	cryptonight_free_ctx(ctx0);
	cryptonight_free_ctx(ctx1);
}

} // namespace cpu
} // namepsace xmrstak
