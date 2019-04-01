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

#include "jconf.hpp"
#include "xmrstak/backend/cpu/cpuType.hpp"
#include "xmrstak/backend/globalStates.hpp"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"

#include "minethd.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/executor.hpp"

#include "hwlocMemory.hpp"
#include "xmrstak/backend/miner_work.hpp"

#ifndef CONF_NO_HWLOC
#include "autoAdjustHwloc.hpp"
#else
#include "autoAdjust.hpp"
#endif

#include <assert.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>

#if defined(__APPLE__)
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#define SYSCTL_CORE_COUNT "machdep.cpu.core_count"
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
	thread_affinity_policy_data_t policy = {static_cast<integer_t>(cpu_id)};
	mach_thread = pthread_mach_thread_np(h);
	return thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1) == KERN_SUCCESS;
#elif defined(__FreeBSD__)
	cpuset_t mn;
	CPU_ZERO(&mn);
	CPU_SET(cpu_id, &mn);
	return pthread_setaffinity_np(h, sizeof(cpuset_t), &mn) == 0;
#elif defined(__OpenBSD__)
	printer::inst()->print_msg(L0, "WARNING: thread pinning is not supported under OPENBSD.");
	return true;
#else
	cpu_set_t mn;
	CPU_ZERO(&mn);
	CPU_SET(cpu_id, &mn);
	return pthread_setaffinity_np(h, sizeof(cpu_set_t), &mn) == 0;
#endif
}

minethd::minethd(miner_work& pWork, size_t iNo, int iMultiway, bool no_prefetch, int64_t affinity, const std::string& asm_version)
{
	this->backendType = iBackend::CPU;
	oWork = pWork;
	bQuit = 0;
	iThreadNo = (uint8_t)iNo;
	iJobNo = 0;
	bNoPrefetch = no_prefetch;
	this->affinity = affinity;
	asm_version_str = asm_version;

	std::unique_lock<std::mutex> lck(thd_aff_set);
	std::future<void> order_guard = order_fix.get_future();

	switch(iMultiway)
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
	alloc_msg msg = {0};

	switch(::jconf::inst()->GetSlowMemSetting())
	{
	case ::jconf::never_use:
		ctx = cryptonight_alloc_ctx(1, 1, &msg);
		if(ctx == NULL)
			printer::inst()->print_msg(L0, "MEMORY ALLOC FAILED: %s", msg.warning);
		else
		{
			ctx->hash_fn = nullptr;
			ctx->loop_fn = nullptr;
			ctx->fun_data = nullptr;
			ctx->asm_version = 0;
			ctx->last_algo = invalid_algo;
		}
		return ctx;

	case ::jconf::no_mlck:
		ctx = cryptonight_alloc_ctx(1, 0, &msg);
		if(ctx == NULL)
			printer::inst()->print_msg(L0, "MEMORY ALLOC FAILED: %s", msg.warning);
		else
		{
			ctx->hash_fn = nullptr;
			ctx->loop_fn = nullptr;
			ctx->fun_data = nullptr;
			ctx->asm_version = 0;
			ctx->last_algo = invalid_algo;
		}
		return ctx;

	case ::jconf::print_warning:
		ctx = cryptonight_alloc_ctx(1, 1, &msg);
		if(msg.warning != NULL)
			printer::inst()->print_msg(L0, "MEMORY ALLOC FAILED: %s", msg.warning);
		if(ctx == NULL)
			ctx = cryptonight_alloc_ctx(0, 0, NULL);

		if(ctx != NULL)
		{
			ctx->hash_fn = nullptr;
			ctx->loop_fn = nullptr;
			ctx->fun_data = nullptr;
			ctx->asm_version = 0;
			ctx->last_algo = invalid_algo;
		}
		return ctx;

	case ::jconf::always_use:
		ctx = cryptonight_alloc_ctx(0, 0, NULL);

		ctx->hash_fn = nullptr;
		ctx->loop_fn = nullptr;
		ctx->fun_data = nullptr;
		ctx->asm_version = 0;
		ctx->last_algo = invalid_algo;

		return ctx;

	case ::jconf::unknown_value:
		return NULL; //Shut up compiler
	}

	return nullptr; //Should never happen
}

static constexpr size_t MAX_N = 5;
bool minethd::self_test()
{
	alloc_msg msg = {0};
	size_t res;
	bool fatal = false;

	switch(::jconf::inst()->GetSlowMemSetting())
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

	cryptonight_ctx* ctx[MAX_N] = {0};
	for(int i = 0; i < MAX_N; i++)
	{
		if((ctx[i] = minethd_alloc_ctx()) == nullptr)
		{
			printer::inst()->print_msg(L0, "ERROR: miner was not able to allocate memory.");
			for(int j = 0; j < i; j++)
				cryptonight_free_ctx(ctx[j]);
			return false;
		}
	}

	bool bResult = true;

	unsigned char out[32 * MAX_N];

	auto neededAlgorithms = ::jconf::inst()->GetCurrentCoinSelection().GetAllAlgorithms();

	for(const auto algo : neededAlgorithms)
	{
		if(algo == POW(cryptonight))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test", 14, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;

			minethd::cn_on_new_job dm;
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test", 14, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05", 32) == 0;

			func_multi_selector<2>(ctx, dm, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("The quick brown fox jumps over the lazy dogThe quick brown fox jumps over the lazy log", 43, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x3e\xbb\x7f\x9f\x7d\x27\x3d\x7c\x31\x8d\x86\x94\x77\x55\x0c\xc8\x00\xcf\xb1\x1b\x0c\xad\xb7\xff\xbd\xf6\xf8\x9f\x3a\x47\x1c\x59"
											 "\xb4\x77\xd5\x02\xe4\xd8\x48\x7f\x42\xdf\xe3\x8e\xed\x73\x81\x7a\xda\x91\xb7\xe2\x63\xd2\x91\x71\xb6\x5c\x44\x3a\x01\x2a\x41\x22",
									 64) == 0;

			func_multi_selector<2>(ctx, dm, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("The quick brown fox jumps over the lazy dogThe quick brown fox jumps over the lazy log", 43, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x3e\xbb\x7f\x9f\x7d\x27\x3d\x7c\x31\x8d\x86\x94\x77\x55\x0c\xc8\x00\xcf\xb1\x1b\x0c\xad\xb7\xff\xbd\xf6\xf8\x9f\x3a\x47\x1c\x59"
											 "\xb4\x77\xd5\x02\xe4\xd8\x48\x7f\x42\xdf\xe3\x8e\xed\x73\x81\x7a\xda\x91\xb7\xe2\x63\xd2\x91\x71\xb6\x5c\x44\x3a\x01\x2a\x41\x22",
									 64) == 0;

			func_multi_selector<3>(ctx, dm, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a testThis is a testThis is a test", 14, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05",
									 96) == 0;

			func_multi_selector<4>(ctx, dm, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a testThis is a testThis is a testThis is a test", 14, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05",
									 128) == 0;

			func_multi_selector<5>(ctx, dm, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a testThis is a testThis is a testThis is a testThis is a test", 14, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05"
											 "\xa0\x84\xf0\x1d\x14\x37\xa0\x9c\x69\x85\x40\x1b\x60\xd4\x35\x54\xae\x10\x58\x02\xc5\xf5\xd8\xa9\xb3\x25\x36\x49\xc0\xbe\x66\x05",
									 160) == 0;
		}
		else if(algo == POW(cryptonight_lite))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x5a\x24\xa0\x29\xde\x1c\x39\x3f\x3d\x52\x7a\x2f\x9b\x39\xdc\x3d\xb3\xbc\x87\x11\x8b\x84\x52\x9b\x9f\x0\x88\x49\x25\x4b\x5\xce", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x5a\x24\xa0\x29\xde\x1c\x39\x3f\x3d\x52\x7a\x2f\x9b\x39\xdc\x3d\xb3\xbc\x87\x11\x8b\x84\x52\x9b\x9f\x0\x88\x49\x25\x4b\x5\xce", 32) == 0;
		}
		else if(algo == POW(cryptonight_monero))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x1\x57\xc5\xee\x18\x8b\xbe\xc8\x97\x52\x85\xa3\x6\x4e\xe9\x20\x65\x21\x76\x72\xfd\x69\xa1\xae\xbd\x7\x66\xc7\xb5\x6e\xe0\xbd", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x1\x57\xc5\xee\x18\x8b\xbe\xc8\x97\x52\x85\xa3\x6\x4e\xe9\x20\x65\x21\x76\x72\xfd\x69\xa1\xae\xbd\x7\x66\xc7\xb5\x6e\xe0\xbd", 32) == 0;
		}
		else if(algo == POW(cryptonight_monero_v8))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = memcmp(out, "\x35\x3f\xdc\x06\x8f\xd4\x7b\x03\xc0\x4b\x94\x31\xe0\x05\xe0\x0b\x68\xc2\x16\x8a\x3c\xc7\x33\x5c\x8b\x9b\x30\x81\x56\x59\x1a\x4f", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult &= memcmp(out, "\x35\x3f\xdc\x06\x8f\xd4\x7b\x03\xc0\x4b\x94\x31\xe0\x05\xe0\x0b\x68\xc2\x16\x8a\x3c\xc7\x33\x5c\x8b\x9b\x30\x81\x56\x59\x1a\x4f", 32) == 0;
		}
		else if(algo == POW(cryptonight_aeon))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xfc\xa1\x7d\x44\x37\x70\x9b\x4a\x3b\xd7\x1e\xf3\xed\x21\xb4\x17\xca\x93\xdc\x86\x79\xce\x81\xdf\xd3\xcb\xdd\xa\x22\xd7\x58\xba", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xfc\xa1\x7d\x44\x37\x70\x9b\x4a\x3b\xd7\x1e\xf3\xed\x21\xb4\x17\xca\x93\xdc\x86\x79\xce\x81\xdf\xd3\xcb\xdd\xa\x22\xd7\x58\xba", 32) == 0;
		}
		else if(algo == POW(cryptonight_ipbc))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xbc\xe7\x48\xaf\xc5\x31\xff\xc9\x33\x7f\xcf\x51\x1b\xe3\x20\xa3\xaa\x8d\x4\x55\xf9\x14\x2a\x61\xe8\x38\xdf\xdc\x3b\x28\x3e\x0xb0", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xbc\xe7\x48\xaf\xc5\x31\xff\xc9\x33\x7f\xcf\x51\x1b\xe3\x20\xa3\xaa\x8d\x4\x55\xf9\x14\x2a\x61\xe8\x38\xdf\xdc\x3b\x28\x3e\x0", 32) == 0;
		}
		else if(algo == POW(cryptonight_stellite))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xb9\x9d\x6c\xee\x50\x3c\x6f\xa6\x3f\x30\x69\x24\x4a\x0\x9f\xe4\xd4\x69\x3f\x68\x92\xa4\x5c\xc2\x51\xae\x46\x87\x7c\x6b\x98\xae", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xb9\x9d\x6c\xee\x50\x3c\x6f\xa6\x3f\x30\x69\x24\x4a\x0\x9f\xe4\xd4\x69\x3f\x68\x92\xa4\x5c\xc2\x51\xae\x46\x87\x7c\x6b\x98\xae", 32) == 0;
		}
		else if(algo == POW(cryptonight_masari))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xbf\x5f\xd\xf3\x5a\x65\x7c\x89\xb0\x41\xcf\xf0\xd\x46\x6a\xb6\x30\xf9\x77\x7f\xd9\xc6\x3\xd7\x3b\xd8\xf1\xb5\x4b\x49\xed\x28", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xbf\x5f\xd\xf3\x5a\x65\x7c\x89\xb0\x41\xcf\xf0\xd\x46\x6a\xb6\x30\xf9\x77\x7f\xd9\xc6\x3\xd7\x3b\xd8\xf1\xb5\x4b\x49\xed\x28", 32) == 0;
		}
		else if(algo == POW(cryptonight_heavy))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xf9\x44\x97\xce\xb4\xf0\xd9\x84\xb\x9b\xfc\x45\x94\x74\x55\x25\xcf\x26\x83\x16\x4f\xc\xf8\x2d\xf5\xf\x25\xff\x45\x28\x2e\x85", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xf9\x44\x97\xce\xb4\xf0\xd9\x84\xb\x9b\xfc\x45\x94\x74\x55\x25\xcf\x26\x83\x16\x4f\xc\xf8\x2d\xf5\xf\x25\xff\x45\x28\x2e\x85", 32) == 0;
		}
		else if(algo == POW(cryptonight_haven))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xc7\xd4\x52\x9\x2b\x48\xa5\xaf\xae\x11\xaf\x40\x9a\x87\xe5\x88\xf0\x29\x35\xa3\x68\xd\xe3\x6b\xce\x43\xf6\xc8\xdf\xd3\xe3\x9", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xc7\xd4\x52\x9\x2b\x48\xa5\xaf\xae\x11\xaf\x40\x9a\x87\xe5\x88\xf0\x29\x35\xa3\x68\xd\xe3\x6b\xce\x43\xf6\xc8\xdf\xd3\xe3\x9", 32) == 0;
		}
		else if(algo == POW(cryptonight_bittube2))
		{
			unsigned char out[32 * MAX_N];
			cn_hash_fun hashf;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);

			ctx[0]->hash_fn("\x38\x27\x4c\x97\xc4\x5a\x17\x2c\xfc\x97\x67\x98\x70\x42\x2e\x3a\x1a\xb0\x78\x49\x60\xc6\x05\x14\xd8\x16\x27\x14\x15\xc3\x06\xee\x3a\x3e\xd1\xa7\x7e\x31\xf6\xa8\x85\xc3\xcb\xff\x01\x02\x03\x04", 48, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x18\x2c\x30\x41\x93\x1a\x14\x73\xc6\xbf\x7e\x77\xfe\xb5\x17\x9b\xa8\xbe\xa9\x68\xba\x9e\xe1\xe8\x24\x1a\x12\x7a\xac\x81\xb4\x24", 32) == 0;

			ctx[0]->hash_fn("\x04\x04\xb4\x94\xce\xd9\x05\x18\xe7\x25\x5d\x01\x28\x63\xde\x8a\x4d\x27\x72\xb1\xff\x78\x8c\xd0\x56\x20\x38\x98\x3e\xd6\x8c\x94\xea\x00\xfe\x43\x66\x68\x83\x00\x00\x00\x00\x18\x7c\x2e\x0f\x66\xf5\x6b\xb9\xef\x67\xed\x35\x14\x5c\x69\xd4\x69\x0d\x1f\x98\x22\x44\x01\x2b\xea\x69\x6e\xe8\xb3\x3c\x42\x12\x01", 76, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x7f\xbe\xb9\x92\x76\x87\x5a\x3c\x43\xc2\xbe\x5a\x73\x36\x06\xb5\xdc\x79\xcc\x9c\xf3\x7c\x43\x3e\xb4\x18\x56\x17\xfb\x9b\xc9\x36", 32) == 0;

			ctx[0]->hash_fn("\x85\x19\xe0\x39\x17\x2b\x0d\x70\xe5\xca\x7b\x33\x83\xd6\xb3\x16\x73\x15\xa4\x22\x74\x7b\x73\xf0\x19\xcf\x95\x28\xf0\xfd\xe3\x41\xfd\x0f\x2a\x63\x03\x0b\xa6\x45\x05\x25\xcf\x6d\xe3\x18\x37\x66\x9a\xf6\xf1\xdf\x81\x31\xfa\xf5\x0a\xaa\xb8\xd3\xa7\x40\x55\x89", 64, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x90\xdc\x65\x53\x8d\xb0\x00\xea\xa2\x52\xcd\xd4\x1c\x17\x7a\x64\xfe\xff\x95\x36\xe7\x71\x68\x35\xd4\xcf\x5c\x73\x56\xb1\x2f\xcd", 32) == 0;
		}
		else if(algo == POW(cryptonight_superfast))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("\x03\x05\xa0\xdb\xd6\xbf\x05\xcf\x16\xe5\x03\xf3\xa6\x6f\x78\x00\x7c\xbf\x34\x14\x43\x32\xec\xbf\xc2\x2e\xd9\x5c\x87\x00\x38\x3b\x30\x9a\xce\x19\x23\xa0\x96\x4b\x00\x00\x00\x08\xba\x93\x9a\x62\x72\x4c\x0d\x75\x81\xfc\xe5\x76\x1e\x9d\x8a\x0e\x6a\x1c\x3f\x92\x4f\xdd\x84\x93\xd1\x11\x56\x49\xc0\x5e\xb6\x01", 76, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x40\x86\x5a\xa8\x87\x41\xec\x1d\xcc\xbd\x2b\xc6\xff\x36\xb9\x4d\x54\x71\x58\xdb\x94\x69\x8e\x3c\xa0\x3d\xe4\x81\x9a\x65\x9f\xef", 32) == 0;
		}
		else if(algo == POW(cryptonight_gpu))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("", 0, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x55\x5e\x0a\xee\x78\x79\x31\x6d\x7d\xef\xf7\x72\x97\x3c\xb9\x11\x8e\x38\x95\x70\x9d\xb2\x54\x7a\xc0\x72\xd5\xb9\x13\x10\x01\xd8", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("", 0, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x55\x5e\x0a\xee\x78\x79\x31\x6d\x7d\xef\xf7\x72\x97\x3c\xb9\x11\x8e\x38\x95\x70\x9d\xb2\x54\x7a\xc0\x72\xd5\xb9\x13\x10\x01\xd8", 32) == 0;
		}
		else if(algo == POW(cryptonight_conceal))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("", 0, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xb5\x54\x4b\x58\x16\x70\x26\x47\x63\x47\xe4\x1f\xb6\x5e\x57\xc9\x7c\xa5\x93\xfe\x0e\xb1\x0f\xb9\x2f\xa7\x3e\x5b\xae\xef\x79\x8c", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("", 0, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xb5\x54\x4b\x58\x16\x70\x26\x47\x63\x47\xe4\x1f\xb6\x5e\x57\xc9\x7c\xa5\x93\xfe\x0e\xb1\x0f\xb9\x2f\xa7\x3e\x5b\xae\xef\x79\x8c", 32) == 0;
		}
		else if(algo == POW(cryptonight_turtle))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x30\x5f\x66\xfe\xbb\xf3\x60\x0e\xda\xbb\x60\xf7\xf1\xc9\xb9\x0a\x3a\xe8\x5a\x31\xd4\x76\xca\x38\x1d\x56\x18\xa6\xc6\x27\x60\xd7", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x30\x5f\x66\xfe\xbb\xf3\x60\x0e\xda\xbb\x60\xf7\xf1\xc9\xb9\x0a\x3a\xe8\x5a\x31\xd4\x76\xca\x38\x1d\x56\x18\xa6\xc6\x27\x60\xd7", 32) == 0;
		}
		else if(algo == POW(cryptonight_r))
		{
			minethd::cn_on_new_job set_job;
			func_multi_selector<1>(ctx, set_job, ::jconf::inst()->HaveHardwareAes(), false, algo);
			miner_work work;
			work.iBlockHeight = 1806260;
			set_job(work, ctx);
			ctx[0]->hash_fn("\x54\x68\x69\x73\x20\x69\x73\x20\x61\x20\x74\x65\x73\x74\x20\x54\x68\x69\x73\x20\x69\x73\x20\x61\x20\x74\x65\x73\x74\x20\x54\x68\x69\x73\x20\x69\x73\x20\x61\x20\x74\x65\x73\x74", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\xf7\x59\x58\x8a\xd5\x7e\x75\x84\x67\x29\x54\x43\xa9\xbd\x71\x49\x0a\xbf\xf8\xe9\xda\xd1\xb9\x5b\x6b\xf2\xf5\xd0\xd7\x83\x87\xbc", 32) == 0;
		}
		else if(algo == POW(cryptonight_v8_reversewaltz))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x32\xf7\x36\xec\x1d\x2f\x3f\xc5\x4c\x49\xbe\xb8\xa0\x47\x6c\xbf\xdd\x14\xc3\x51\xb9\xc6\xd7\x2c\x6f\x9f\xfc\xb5\x87\x5b\xe6\xb3", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x32\xf7\x36\xec\x1d\x2f\x3f\xc5\x4c\x49\xbe\xb8\xa0\x47\x6c\xbf\xdd\x14\xc3\x51\xb9\xc6\xd7\x2c\x6f\x9f\xfc\xb5\x87\x5b\xe6\xb3", 32) == 0;
		}
		else if(algo == POW(cryptonight_v8_double))
		{
			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), false, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x63\x43\x8e\xd\x5c\x18\xff\xca\xd5\xb5\xdf\xe0\x26\x8a\x5b\x3f\xe9\xbc\x1\xef\xe6\x3a\xd3\x4f\x2c\x57\x1c\xda\xb2\xc\x32\x31", 32) == 0;

			func_selector(ctx, ::jconf::inst()->HaveHardwareAes(), true, algo);
			ctx[0]->hash_fn("This is a test This is a test This is a test", 44, out, ctx, algo);
			bResult = bResult && memcmp(out, "\x63\x43\x8e\xd\x5c\x18\xff\xca\xd5\xb5\xdf\xe0\x26\x8a\x5b\x3f\xe9\xbc\x1\xef\xe6\x3a\xd3\x4f\x2c\x57\x1c\xda\xb2\xc\x32\x31", 32) == 0;
		}
		else
			printer::inst()->print_msg(L0,
				"Cryptonight hash self-test NOT defined for POW %s", algo.Name().c_str());

		if(!bResult)
			printer::inst()->print_msg(L0,
				"Cryptonight hash self-test failed. This might be caused by bad compiler optimizations.");
	}

	for(int i = 0; i < MAX_N; i++)
		cryptonight_free_ctx(ctx[i]);

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
	for(i = 0; i < n; i++)
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

		minethd* thd = new minethd(pWork, i + threadOffset, cfg.iMultiway, cfg.bNoPrefetch, cfg.iCpuAff, cfg.asm_version_str);
		pvThreads.push_back(thd);
	}

	return pvThreads;
}

/** get the supported asm name
 *
 * @return asm type based on the number of hashes per thread the internal
 *             evaluated cpu type
 */
static std::string getAsmName(const uint32_t num_hashes)
{
	std::string asm_type = "off";
	if(num_hashes != 0)
	{
		auto cpu_model = getModel();

		if(cpu_model.avx && cpu_model.aes)
		{
			if(cpu_model.type_name.find("Intel") != std::string::npos)
				asm_type = "intel_avx";
			else if(cpu_model.type_name.find("AMD") != std::string::npos)
				asm_type = "amd_avx";
		}
	}
	return asm_type;
}

template <size_t N>
void minethd::func_multi_selector(cryptonight_ctx** ctx, minethd::cn_on_new_job& on_new_job,
	bool bHaveAes, bool bNoPrefetch, const xmrstak_algo& algo, const std::string& asm_version_str)
{
	static_assert(N >= 1, "number of threads must be >= 1");

	// We have two independent flag bits in the functions
	// therefore we will build a binary digit and select the
	// function as a two digit binary

	uint8_t algv;
	switch(algo.Id())
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
	case cryptonight_ipbc:
		algv = 5;
		break;
	case cryptonight_stellite:
		algv = 6;
		break;
	case cryptonight_masari:
		algv = 7;
		break;
	case cryptonight_haven:
		algv = 8;
		break;
	case cryptonight_bittube2:
		algv = 9;
		break;
	case cryptonight_monero_v8:
		algv = 10;
		break;
	case cryptonight_superfast:
		algv = 11;
		break;
	case cryptonight_gpu:
		algv = 12;
		break;
	case cryptonight_conceal:
		algv = 13;
		break;
	case cryptonight_r:
		algv = 14;
		break;
	case cryptonight_v8_reversewaltz:
		algv = 15;
		break;
	default:
		algv = 2;
		break;
	}

	static const cn_hash_fun func_table[] = {
		Cryptonight_hash<N>::template hash<cryptonight_monero, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_monero, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_monero, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_monero, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_lite, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_lite, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_lite, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_lite, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_heavy, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_heavy, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_heavy, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_heavy, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_aeon, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_aeon, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_aeon, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_aeon, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_ipbc, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_ipbc, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_ipbc, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_ipbc, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_stellite, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_stellite, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_stellite, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_stellite, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_masari, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_masari, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_masari, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_masari, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_haven, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_haven, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_haven, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_haven, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_bittube2, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_bittube2, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_bittube2, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_bittube2, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_monero_v8, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_monero_v8, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_monero_v8, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_monero_v8, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_superfast, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_superfast, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_superfast, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_superfast, true, true>,

		Cryptonight_hash_gpu::template hash<cryptonight_gpu, false, false>,
		Cryptonight_hash_gpu::template hash<cryptonight_gpu, true, false>,
		Cryptonight_hash_gpu::template hash<cryptonight_gpu, false, true>,
		Cryptonight_hash_gpu::template hash<cryptonight_gpu, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_conceal, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_conceal, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_conceal, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_conceal, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_r, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_r, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_r, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_r, true, true>,

		Cryptonight_hash<N>::template hash<cryptonight_v8_reversewaltz, false, false>,
		Cryptonight_hash<N>::template hash<cryptonight_v8_reversewaltz, true, false>,
		Cryptonight_hash<N>::template hash<cryptonight_v8_reversewaltz, false, true>,
		Cryptonight_hash<N>::template hash<cryptonight_v8_reversewaltz, true, true>};

	std::bitset<2> digit;
	digit.set(0, !bHaveAes);
	digit.set(1, !bNoPrefetch);

	ctx[0]->hash_fn = func_table[algv << 2 | digit.to_ulong()];

	// check for asm optimized version for cryptonight_v8
	if(algo == cryptonight_monero_v8)
	{
		std::string selected_asm = asm_version_str;
		if(selected_asm == "auto")
			selected_asm = cpu::getAsmName(N);

		if(selected_asm != "off")
		{
			patchAsmVariants<N>(selected_asm, ctx, algo);

			if(asm_version_str == "auto" && (selected_asm != "intel_avx" || selected_asm != "amd_avx"))
				printer::inst()->print_msg(L3, "Switch to assembler version for '%s' cpu's", selected_asm.c_str());
			else if(selected_asm != "intel_avx" && selected_asm != "amd_avx") // unknown asm type
				printer::inst()->print_msg(L1, "Assembler '%s' unknown, fallback to non asm version of cryptonight_v8", selected_asm.c_str());
		}
	}
	else if(algo == cryptonight_r && asm_version_str != "off")
	{
		std::string selected_asm = asm_version_str;
		if(selected_asm == "auto")
			selected_asm = cpu::getAsmName(N);
		if(selected_asm == "off")
		{
			for(int h = 0; h < N; ++h)
				ctx[h]->asm_version = 0;
		}
		else
		{
			printer::inst()->print_msg(L0, "enable cryptonight_r asm '%s' cpu's", selected_asm.c_str());
			for(int h = 0; h < N; ++h)
				ctx[h]->asm_version = selected_asm == "intel_avx" ? 1 : 2; // 1 == Intel; 2 == AMD
		}
	}

	for(int h = 1; h < N; ++h)
		ctx[h]->hash_fn = ctx[0]->hash_fn;

	static const std::unordered_map<uint32_t, minethd::cn_on_new_job> on_new_job_map = {
		{cryptonight_r, Cryptonight_R_generator<N>::template cn_on_new_job<cryptonight_r>},
	};

	auto it = on_new_job_map.find(algo.Id());
	if(it != on_new_job_map.end())
		on_new_job = it->second;
	else
		on_new_job = nullptr;
}

void minethd::func_selector(cryptonight_ctx** ctx, bool bHaveAes, bool bNoPrefetch, const xmrstak_algo& algo)
{
	minethd::cn_on_new_job dm;
	func_multi_selector<1>(ctx, dm, bHaveAes, bNoPrefetch, algo); // for testing us eauto, must be removed before the release
}

void minethd::work_main()
{
	multiway_work_main<1u>();
}

void minethd::double_work_main()
{
	multiway_work_main<2u>();
}

void minethd::triple_work_main()
{
	multiway_work_main<3u>();
}

void minethd::quad_work_main()
{
	multiway_work_main<4u>();
}

void minethd::penta_work_main()
{
	multiway_work_main<5u>();
}

template <size_t N>
void minethd::prep_multiway_work(uint8_t* bWorkBlob, uint32_t** piNonce)
{
	for(size_t i = 0; i < N; i++)
	{
		memcpy(bWorkBlob + oWork.iWorkSize * i, oWork.bWorkBlob, oWork.iWorkSize);
		if(i > 0)
			piNonce[i] = (uint32_t*)(bWorkBlob + oWork.iWorkSize * i + 39);
	}
}

template <uint32_t N>
void minethd::multiway_work_main()
{
	if(affinity >= 0) //-1 means no affinity
		bindMemoryToNUMANode(affinity);

	order_fix.set_value();
	std::unique_lock<std::mutex> lck(thd_aff_set);
	lck.release();
	std::this_thread::yield();

	cryptonight_ctx* ctx[MAX_N];
	uint64_t iCount = 0;
	uint64_t iLastCount = 0;
	uint64_t* piHashVal[MAX_N];
	uint32_t* piNonce[MAX_N];
	uint8_t bHashOut[MAX_N * 32];
	uint8_t bWorkBlob[sizeof(miner_work::bWorkBlob) * MAX_N];
	uint32_t iNonce;
	job_result res;

	for(size_t i = 0; i < N; i++)
	{
		ctx[i] = minethd_alloc_ctx();
		if(ctx[i] == nullptr)
		{
			printer::inst()->print_msg(L0, "ERROR: miner was not able to allocate memory.");
			for(int j = 0; j < i; j++)
				cryptonight_free_ctx(ctx[j]);
			win_exit(1);
		}
		piHashVal[i] = (uint64_t*)(bHashOut + 32 * i + 24);
		piNonce[i] = (i == 0) ? (uint32_t*)(bWorkBlob + 39) : nullptr;
	}

	if(!oWork.bStall)
		prep_multiway_work<N>(bWorkBlob, piNonce);

	globalStates::inst().iConsumeCnt++;

	// start with root algorithm and switch later if fork version is reached
	auto miner_algo = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot();
	cn_on_new_job on_new_job;
	uint8_t version = 0;
	size_t lastPoolId = 0;

	func_multi_selector<N>(ctx, on_new_job, ::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo, asm_version_str);
	while(bQuit == 0)
	{
		if(oWork.bStall)
		{
			/*	We are stalled here because the executor didn't find a job for us yet,
			either because of network latency, or a socket problem. Since we are
			raison d'etre of this software it us sensible to just wait until we have something*/

			while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
				std::this_thread::sleep_for(std::chrono::milliseconds(100));

			globalStates::inst().consume_work(oWork, iJobNo);
			prep_multiway_work<N>(bWorkBlob, piNonce);
			continue;
		}

		constexpr uint32_t nonce_chunk = 4096;
		int64_t nonce_ctr = 0;

		assert(sizeof(job_result::sJobID) == sizeof(pool_job::sJobID));

		if(oWork.bNiceHash)
			iNonce = *piNonce[0];

		uint8_t new_version = oWork.getVersion();
		if(new_version != version || oWork.iPoolId != lastPoolId)
		{
			coinDescription coinDesc = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(oWork.iPoolId);
			if(new_version >= coinDesc.GetMiningForkVersion())
			{
				miner_algo = coinDesc.GetMiningAlgo();
				func_multi_selector<N>(ctx, on_new_job, ::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo, asm_version_str);
			}
			else
			{
				miner_algo = coinDesc.GetMiningAlgoRoot();
				func_multi_selector<N>(ctx, on_new_job, ::jconf::inst()->HaveHardwareAes(), bNoPrefetch, miner_algo, asm_version_str);
			}
			lastPoolId = oWork.iPoolId;
			version = new_version;
		}

		if(on_new_job != nullptr)
			on_new_job(oWork, ctx);

		while(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) == iJobNo)
		{
			if((iCount++ & 0x7) == 0) //Store stats every 8*N hashes
			{
				updateStats((iCount - iLastCount) * N, oWork.iPoolId);
				iLastCount = iCount;
			}

			nonce_ctr -= N;
			if(nonce_ctr <= 0)
			{
				globalStates::inst().calc_start_nonce(iNonce, oWork.bNiceHash, nonce_chunk);
				nonce_ctr = nonce_chunk;
				// check if the job is still valid, there is a small posibility that the job is switched
				if(globalStates::inst().iGlobalJobNo.load(std::memory_order_relaxed) != iJobNo)
					break;
			}

			for(size_t i = 0; i < N; i++)
				*piNonce[i] = iNonce++;

			ctx[0]->hash_fn(bWorkBlob, oWork.iWorkSize, bHashOut, ctx, miner_algo);

			for(size_t i = 0; i < N; i++)
			{
				if(*piHashVal[i] < oWork.iTarget)
				{
					executor::inst()->push_event(
						ex_event(job_result(oWork.sJobID, iNonce - N + i, bHashOut + 32 * i, iThreadNo, miner_algo),
							oWork.iPoolId));
				}
			}

			std::this_thread::yield();
		}

		globalStates::inst().consume_work(oWork, iJobNo);
		prep_multiway_work<N>(bWorkBlob, piNonce);
	}

	for(int i = 0; i < N; i++)
		cryptonight_free_ctx(ctx[i]);
}

} // namespace cpu
} // namespace xmrstak
