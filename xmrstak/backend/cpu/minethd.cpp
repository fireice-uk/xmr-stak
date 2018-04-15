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
	// we can only pin up to 64 threads
	if(cpu_id < 64)
	{
		return SetThreadAffinityMask(h, 1ULL << cpu_id) != 0;
	}
	else
	{
		printer::inst()->print_msg(L0, "WARNING: Windows supports only affinity up to 63.");
		return false;
	}
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
#elif defined(__OpenBSD__)
        printer::inst()->print_msg(L0,"WARNING: thread pinning is not supported under OPENBSD.");
#else
	cpu_set_t mn;
	CPU_ZERO(&mn);
	CPU_SET(cpu_id, &mn);
	return pthread_setaffinity_np(h, sizeof(cpu_set_t), &mn) == 0;
#endif
}

minethd::minethd(miner_work& pWork, size_t iNo, int iMultiway, bool no_prefetch, int64_t affinity)
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

	switch (iMultiway)
	{
	case 5:
		oWorkThd = std::thread(&minethd::penta_work_main, this);
		break;
	case 4:
		oWorkThd = std::thread(&minethd::quad_work_main, this);
		break;
	case 3:
		oWorkThd = std::thread(&minethd::triple_work_main, this);
		break;
	case 2:
		oWorkThd = std::thread(&minethd::double_work_main, this);
		break;
	case 1:
	default:
		oWorkThd = std::thread(&minethd::work_main, this);
		break;
	}

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

static constexpr size_t MAX_N = 5;
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

	cryptonight_ctx *ctx[MAX_N] = {0};
	for (int i = 0; i < MAX_N; i++)
	{
		if ((ctx[i] = minethd_alloc_ctx()) == nullptr)
		{
			for (int j = 0; j < i; j++)
				cryptonight_free_ctx(ctx[j]);
			return false;
		}
	}

	bool bResult = true;

	if(::jconf::inst()->GetMiningAlgo() == cryptonight)
	{
		unsigned char out[32 * MAX_N];
		cn_hash_fun hashf;
		cn_hash_fun_multi hashf_multi;

		hashf = func_selector(::jconf::inst()->HaveHardwareAes(), false, xmrstak_algo::cryptonight);
		hashf("This is a test", 14, out, ctx[0]);
		bResult = memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;

		hashf = func_selector(::jconf::inst()->HaveHardwareAes(), true, xmrstak_algo::cryptonight);
		hashf("This is a test", 14, out, ctx[0]);
		bResult &= memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;

		hashf_multi = func_multi_selector(2, ::jconf::inst()->HaveHardwareAes(), false, xmrstak_algo::cryptonight);
		hashf_multi("The quick brown fox jumps over the lazy dogThe quick brown fox jumps over the lazy log", 43, out, ctx);
		bResult &= memcmp(out, "\x3e\xbb\x7f\x9f\x7d\x27\x3d\x7c\x31\x8d\x86\x94\x77\x55\x0c\xc8\x00\xcf\xb1\x1b\x0c\xad\xb7\xff\xbd\xf6\xf8\x9f\x3a\x47\x1c\x59"
				"\xb4\x77\xd5\x02\xe4\xd8\x48\x7f\x42\xdf\xe3\x8e\xed\x73\x81\x7a\xda\x91\xb7\xe2\x63\xd2\x91\x71\xb6\x5c\x44\x3a\x01\x2a\x41\x22", 64) == 0;

		hashf_multi = func_multi_selector(2, ::jconf::inst()->HaveHardwareAes(), true, xmrstak_algo::cryptonight);
		hashf_multi("The quick brown fox jumps over the lazy dogThe quick brown fox jumps over the lazy log", 43, out, ctx);
		bResult &= memcmp(out, "\x3e\xbb\x7f\x9f\x7d\x27\x3d\x7c\x31\x8d\x86\x94\x77\x55\x0c\xc8\x00\xcf\xb1\x1b\x0c\xad\xb7\xff\xbd\xf6\xf8\x9f\x3a\x47\x1c\x59"
				"\xb4\x77\xd5\x02\xe4\xd8\x48\x7f\x42\xdf\xe3\x8e\xed\x73\x81\x7a\xda\x91\xb7\xe2\x63\xd2\x91\x71\xb6\x5c\x44\x3a\x01\x2a\x41\x22", 64) == 0;

		hashf_multi = func_multi_selector(3, ::jconf::inst()->HaveHardwareAes(), false, xmrstak_algo::cryptonight);
		hashf_multi("This is a testThis is a testThis is a test", 14, out, ctx);
		bResult &= memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 96) == 0;

		hashf_multi = func_multi_selector(4, ::jconf::inst()->HaveHardwareAes(), false, xmrstak_algo::cryptonight);
		hashf_multi("This is a testThis is a testThis is a testThis is a test", 14, out, ctx);
		bResult &= memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 128) == 0;

		hashf_multi = func_multi_selector(5, ::jconf::inst()->HaveHardwareAes(), false, xmrstak_algo::cryptonight);
		hashf_multi("This is a testThis is a testThis is a testThis is a testThis is a test", 14, out, ctx);
		bResult &= memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
				"\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 160) == 0;
	}
	else if(::jconf::inst()->GetMiningAlgo() == cryptonight_lite)
	{
	}
	else if(::jconf::inst()->GetMiningAlgo() == cryptonight_monero)
	{
	}
	else if(::jconf::inst()->GetMiningAlgo() == cryptonight_aeon)
	{
	}

	for (int i = 0; i < MAX_N; i++)
		cryptonight_free_ctx(ctx[i]);

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
			printer::inst()->print_msg(L1, "WARNING on macOS thread affinity is only advisory.");
#endif

			printer::inst()->print_msg(L1, "Starting %dx thread, affinity: %d.", cfg.iMultiway, (int)cfg.iCpuAff);
		}
		else
			printer::inst()->print_msg(L1, "Starting %dx thread, no affinity.", cfg.iMultiway);
		
		minethd* thd = new minethd(pWork, i + threadOffset, cfg.iMultiway, cfg.bNoPrefetch, cfg.iCpuAff);
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

minethd::cn_hash_fun minethd::func_selector(bool bHaveAes, bool bNoPrefetch, xmrstak_algo algo)
{
	// We have two independent flag bits in the functions
	// therefore we will build a binary digit and select the
	// function as a two digit binary

	uint8_t algv;
	switch(algo)
	{
	case cryptonight:
		algv = 2;
		break;
	case cryptonight_lite:
		algv = 1;
		break;
	case cryptonight_monero:
		algv = 0;
		break;
	case cryptonight_heavy:
		algv = 3;
		break;
	case cryptonight_aeon:
		algv = 4;
		break;
	default:
		algv = 2;
		break;
	}

	static const cn_hash_fun func_table[] = {
		cryptonight_hash<cryptonight_monero, false, false>,
		cryptonight_hash<cryptonight_monero, true, false>,
		cryptonight_hash<cryptonight_monero, false, true>,
		cryptonight_hash<cryptonight_monero, true, true>,
		cryptonight_hash<cryptonight_lite, false, false>,
		cryptonight_hash<cryptonight_lite, true, false>,
		cryptonight_hash<cryptonight_lite, false, true>,
		cryptonight_hash<cryptonight_lite, true, true>,
		cryptonight_hash<cryptonight, false, false>,
		cryptonight_hash<cryptonight, true, false>,
		cryptonight_hash<cryptonight, false, true>,
		cryptonight_hash<cryptonight, true, true>,
		cryptonight_hash<cryptonight_heavy, false, false>,
		cryptonight_hash<cryptonight_heavy, true, false>,
		cryptonight_hash<cryptonight_heavy, false, true>,
		cryptonight_hash<cryptonight_heavy, true, true>,
		cryptonight_hash<cryptonight_aeon, false, false>,
		cryptonight_hash<cryptonight_aeon, true, false>,
		cryptonight_hash<cryptonight_aeon, false, true>,
		cryptonight_hash<cryptonight_aeon, true, true>
	};

	std::bitset<2> digit;
	digit.set(0, !bHaveAes);
	digit.set(1, !bNoPrefetch);

	return func_table[ algv << 2 | digit.to_ulong() ];
}

void minethd::work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.release();
	std::this_thread::yield();

	cryptonight_ctx* ctx;
	uint64_t iCount = 0;
	uint64_t* piHashVal;
	uint32_t* piNonce;
	job_result result;

	// start with root algorithm and switch later if fork version is reached
	auto miner_algo = ::jconf::inst()->GetMiningAlgoRoot();
	cn_hash_fun hash_fun = func_selector(::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo);
	ctx = minethd_alloc_ctx();

	piHashVal = (uint64_t*)(result.bResult + 24);
	piNonce = (uint32_t*)(oWork.bWorkBlob + 39);
	globalStates::inst().inst().iConsumeCnt++;
	result.iThreadId = iThreadNo;

	uint8_t version = 0;

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

		uint8_t new_version = oWork.getVersion();
		if(new_version != version)
		{
			if(new_version >= ::jconf::inst()->GetMiningForkVersion())
			{
				miner_algo = ::jconf::inst()->GetMiningAlgo();
				hash_fun = func_selector(::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo);
			}
			version = new_version;
		}

		while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			if ((iCount++ & 0xF) == 0) //Store stats every 16 hashes
			{
				uint64_t iStamp = get_timestamp_ms();
				iHashCount.store(iCount, std::memory_order_relaxed);
				iTimestamp.store(iStamp, std::memory_order_relaxed);
			}

			if((nonce_ctr++ & (nonce_chunk-1)) == 0)
			{
				globalStates::inst().calc_start_nonce(result.iNonce, oWork.bNiceHash, nonce_chunk);
			}

			*piNonce = result.iNonce;

			hash_fun(oWork.bWorkBlob, oWork.iWorkSize, result.bResult, ctx);

			if (*piHashVal < oWork.iTarget)
				executor::inst()->push_event(ex_event(result, oWork.iPoolId));
			result.iNonce++;

			std::this_thread::yield();
		}

		consume_work();
	}

	cryptonight_free_ctx(ctx);
}

minethd::cn_hash_fun_multi minethd::func_multi_selector(size_t N, bool bHaveAes, bool bNoPrefetch, xmrstak_algo algo)
{
	// We have two independent flag bits in the functions
	// therefore we will build a binary digit and select the
	// function as a two digit binary

	uint8_t algv;
	switch(algo)
	{
	case cryptonight:
		algv = 2;
		break;
	case cryptonight_lite:
		algv = 1;
		break;
	case cryptonight_monero:
		algv = 0;
		break;
	case cryptonight_heavy:
		algv = 3;
		break;
	case cryptonight_aeon:
		algv = 4;
		break;
	default:
		algv = 2;
		break;
	}

	static const cn_hash_fun_multi func_table[] = {
		cryptonight_double_hash<cryptonight_monero, false, false>,
		cryptonight_double_hash<cryptonight_monero, true, false>,
		cryptonight_double_hash<cryptonight_monero, false, true>,
		cryptonight_double_hash<cryptonight_monero, true, true>,
		cryptonight_triple_hash<cryptonight_monero, false, false>,
		cryptonight_triple_hash<cryptonight_monero, true, false>,
		cryptonight_triple_hash<cryptonight_monero, false, true>,
		cryptonight_triple_hash<cryptonight_monero, true, true>,
		cryptonight_quad_hash<cryptonight_monero, false, false>,
		cryptonight_quad_hash<cryptonight_monero, true, false>,
		cryptonight_quad_hash<cryptonight_monero, false, true>,
		cryptonight_quad_hash<cryptonight_monero, true, true>,
		cryptonight_penta_hash<cryptonight_monero, false, false>,
		cryptonight_penta_hash<cryptonight_monero, true, false>,
		cryptonight_penta_hash<cryptonight_monero, false, true>,
		cryptonight_penta_hash<cryptonight_monero, true, true>,

		cryptonight_double_hash<cryptonight_lite, false, false>,
		cryptonight_double_hash<cryptonight_lite, true, false>,
		cryptonight_double_hash<cryptonight_lite, false, true>,
		cryptonight_double_hash<cryptonight_lite, true, true>,
		cryptonight_triple_hash<cryptonight_lite, false, false>,
		cryptonight_triple_hash<cryptonight_lite, true, false>,
		cryptonight_triple_hash<cryptonight_lite, false, true>,
		cryptonight_triple_hash<cryptonight_lite, true, true>,
		cryptonight_quad_hash<cryptonight_lite, false, false>,
		cryptonight_quad_hash<cryptonight_lite, true, false>,
		cryptonight_quad_hash<cryptonight_lite, false, true>,
		cryptonight_quad_hash<cryptonight_lite, true, true>,
		cryptonight_penta_hash<cryptonight_lite, false, false>,
		cryptonight_penta_hash<cryptonight_lite, true, false>,
		cryptonight_penta_hash<cryptonight_lite, false, true>,
		cryptonight_penta_hash<cryptonight_lite, true, true>,
		
		cryptonight_double_hash<cryptonight, false, false>,
		cryptonight_double_hash<cryptonight, true, false>,
		cryptonight_double_hash<cryptonight, false, true>,
		cryptonight_double_hash<cryptonight, true, true>,
		cryptonight_triple_hash<cryptonight, false, false>,
		cryptonight_triple_hash<cryptonight, true, false>,
		cryptonight_triple_hash<cryptonight, false, true>,
		cryptonight_triple_hash<cryptonight, true, true>,
		cryptonight_quad_hash<cryptonight, false, false>,
		cryptonight_quad_hash<cryptonight, true, false>,
		cryptonight_quad_hash<cryptonight, false, true>,
		cryptonight_quad_hash<cryptonight, true, true>,
		cryptonight_penta_hash<cryptonight, false, false>,
		cryptonight_penta_hash<cryptonight, true, false>,
		cryptonight_penta_hash<cryptonight, false, true>,
		cryptonight_penta_hash<cryptonight, true, true>,

		cryptonight_double_hash<cryptonight_heavy, false, false>,
		cryptonight_double_hash<cryptonight_heavy, true, false>,
		cryptonight_double_hash<cryptonight_heavy, false, true>,
		cryptonight_double_hash<cryptonight_heavy, true, true>,
		cryptonight_triple_hash<cryptonight_heavy, false, false>,
		cryptonight_triple_hash<cryptonight_heavy, true, false>,
		cryptonight_triple_hash<cryptonight_heavy, false, true>,
		cryptonight_triple_hash<cryptonight_heavy, true, true>,
		cryptonight_quad_hash<cryptonight_heavy, false, false>,
		cryptonight_quad_hash<cryptonight_heavy, true, false>,
		cryptonight_quad_hash<cryptonight_heavy, false, true>,
		cryptonight_quad_hash<cryptonight_heavy, true, true>,
		cryptonight_penta_hash<cryptonight_heavy, false, false>,
		cryptonight_penta_hash<cryptonight_heavy, true, false>,
		cryptonight_penta_hash<cryptonight_heavy, false, true>,
		cryptonight_penta_hash<cryptonight_heavy, true, true>,

		cryptonight_double_hash<cryptonight_aeon, false, false>,
		cryptonight_double_hash<cryptonight_aeon, true, false>,
		cryptonight_double_hash<cryptonight_aeon, false, true>,
		cryptonight_double_hash<cryptonight_aeon, true, true>,
		cryptonight_triple_hash<cryptonight_aeon, false, false>,
		cryptonight_triple_hash<cryptonight_aeon, true, false>,
		cryptonight_triple_hash<cryptonight_aeon, false, true>,
		cryptonight_triple_hash<cryptonight_aeon, true, true>,
		cryptonight_quad_hash<cryptonight_aeon, false, false>,
		cryptonight_quad_hash<cryptonight_aeon, true, false>,
		cryptonight_quad_hash<cryptonight_aeon, false, true>,
		cryptonight_quad_hash<cryptonight_aeon, true, true>,
		cryptonight_penta_hash<cryptonight_aeon, false, false>,
		cryptonight_penta_hash<cryptonight_aeon, true, false>,
		cryptonight_penta_hash<cryptonight_aeon, false, true>,
		cryptonight_penta_hash<cryptonight_aeon, true, true>
	};

	std::bitset<2> digit;
	digit.set(0, !bHaveAes);
	digit.set(1, !bNoPrefetch);
	
	return func_table[algv << 4 | (N-2) << 2 | digit.to_ulong()];
}

void minethd::double_work_main()
{
	multiway_work_main<2>();
}

void minethd::triple_work_main()
{
	multiway_work_main<3>();
}

void minethd::quad_work_main()
{
	multiway_work_main<4>();
}

void minethd::penta_work_main()
{
	multiway_work_main<5>();
}

template<size_t N>
void minethd::prep_multiway_work(uint8_t *bWorkBlob, uint32_t **piNonce)
{
	for (size_t i = 0; i < N; i++)
	{
		memcpy(bWorkBlob + oWork.iWorkSize * i, oWork.bWorkBlob, oWork.iWorkSize);
		if (i > 0)
			piNonce[i] = (uint32_t*)(bWorkBlob + oWork.iWorkSize * i + 39);
	}
}

template<size_t N>
void minethd::multiway_work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.release();
	std::this_thread::yield();

	cryptonight_ctx *ctx[MAX_N];
	uint64_t iCount = 0;
	uint64_t *piHashVal[MAX_N];
	uint32_t *piNonce[MAX_N];
	uint8_t bHashOut[MAX_N * 32];
	uint8_t bWorkBlob[sizeof(miner_work::bWorkBlob) * MAX_N];
	uint32_t iNonce;
	job_result res;

	for (size_t i = 0; i < N; i++)
	{
		ctx[i] = minethd_alloc_ctx();
		piHashVal[i] = (uint64_t*)(bHashOut + 32 * i + 24);
		piNonce[i] = (i == 0) ? (uint32_t*)(bWorkBlob + 39) : nullptr;
	}

	if(!oWork.bStall)
		prep_multiway_work<N>(bWorkBlob, piNonce);

	globalStates::inst().iConsumeCnt++;

	// start with root algorithm and switch later if fork version is reached
	auto miner_algo = ::jconf::inst()->GetMiningAlgoRoot();
	cn_hash_fun_multi hash_fun_multi = func_multi_selector(N, ::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo);
	uint8_t version = 0;

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
			prep_multiway_work<N>(bWorkBlob, piNonce);
			continue;
		}

		constexpr uint32_t nonce_chunk = 4096;
		int64_t nonce_ctr = 0;

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));

		if(oWork.bNiceHash)
			iNonce = *piNonce[0];

		uint8_t new_version = oWork.getVersion();
		if(new_version != version)
		{
			if(new_version >= ::jconf::inst()->GetMiningForkVersion())
			{
				miner_algo = ::jconf::inst()->GetMiningAlgo();
				hash_fun_multi = func_multi_selector(N, ::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo);
			}
			version = new_version;
		}

		while (globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			if ((iCount++ & 0x7) == 0)  //Store stats every 8*N hashes
			{
				uint64_t iStamp = get_timestamp_ms();
				iHashCount.store(iCount * N, std::memory_order_relaxed);
				iTimestamp.store(iStamp, std::memory_order_relaxed);
			}

			nonce_ctr -= N;
			if(nonce_ctr <= 0)
			{
				globalStates::inst().calc_start_nonce(iNonce, oWork.bNiceHash, nonce_chunk);
				nonce_ctr = nonce_chunk;
			}

			for (size_t i = 0; i < N; i++)
				*piNonce[i] = iNonce++;

			hash_fun_multi(bWorkBlob, oWork.iWorkSize, bHashOut, ctx);

			for (size_t i = 0; i < N; i++)
			{
				if (*piHashVal[i] < oWork.iTarget)
				{
					executor::inst()->push_event(ex_event(job_result(oWork.sJobID, iNonce - N + i, bHashOut + 32 * i, iThreadNo), oWork.iPoolId));
				}
			}

			std::this_thread::yield();
		}

		consume_work();
		prep_multiway_work<N>(bWorkBlob, piNonce);
	}

	for (int i = 0; i < N; i++)
		cryptonight_free_ctx(ctx[i]);
}

} // namespace cpu
} // namespace xmrstak
