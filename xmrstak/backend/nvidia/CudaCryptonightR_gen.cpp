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
 */

#include <cstring>
#include <mutex>
#include <nvrtc.h>
#include <sstream>
#include <string>
#include <thread>

#include "xmrstak/backend/cpu/crypto/variant4_random_math.h"
#include "xmrstak/backend/nvidia/CudaCryptonightR_gen.hpp"
#include "xmrstak/cpputil/read_write_lock.h"
#include "xmrstak/misc/console.hpp"

namespace xmrstak
{
namespace nvidia
{

static std::string get_code(const V4_Instruction* code, int code_size)
{
	std::stringstream s;

	for(int i = 0; i < code_size; ++i)
	{
		const V4_Instruction inst = code[i];

		const uint32_t a = inst.dst_index;
		const uint32_t b = inst.src_index;

		switch(inst.opcode)
		{
		case MUL:
			s << 'r' << a << "*=r" << b << ';';
			break;

		case ADD:
			s << 'r' << a << "+=r" << b << '+' << inst.C << "U;";
			break;

		case SUB:
			s << 'r' << a << "-=r" << b << ';';
			break;

		case ROR:
			s << 'r' << a << "=rotate_right(r" << a << ",r" << b << ");";
			break;

		case ROL:
			s << 'r' << a << "=rotate_left(r" << a << ",r" << b << ");";
			break;

		case XOR:
			s << 'r' << a << "^=r" << b << ';';
			break;
		}

		s << '\n';
	}

	return s.str();
}

struct CacheEntry
{
	CacheEntry(xmrstak_algo algo, uint64_t height, int arch_major, int arch_minor, const std::vector<char>& ptx, const std::string& lowered_name) :
		algo(algo),
		height(height),
		arch_major(arch_major),
		arch_minor(arch_minor),
		ptx(ptx),
		lowered_name(lowered_name)
	{
	}

	xmrstak_algo algo;
	uint64_t height;
	int arch_major;
	int arch_minor;
	std::vector<char> ptx;
	std::string lowered_name;
};

struct BackgroundTaskBase
{
	virtual ~BackgroundTaskBase() {}
	virtual void exec() = 0;
};

template <typename T>
struct BackgroundTask : public BackgroundTaskBase
{
	BackgroundTask(T&& func) :
		m_func(std::move(func)) {}
	void exec() override { m_func(); }

	T m_func;
};

static ::cpputil::RWLock CryptonightR_cache_mutex;
static std::mutex CryptonightR_build_mutex;
static std::vector<CacheEntry> CryptonightR_cache;

static std::mutex background_tasks_mutex;
static std::vector<BackgroundTaskBase*> background_tasks;
static std::thread* background_thread = nullptr;

static void background_thread_proc()
{
	std::vector<BackgroundTaskBase*> tasks;
	for(;;)
	{
		tasks.clear();
		{
			std::lock_guard<std::mutex> g(background_tasks_mutex);
			background_tasks.swap(tasks);
		}

		for(BackgroundTaskBase* task : tasks)
		{
			task->exec();
			delete task;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
}

template <typename T>
static void background_exec(T&& func)
{
	BackgroundTaskBase* task = new BackgroundTask<T>(std::move(func));

	std::lock_guard<std::mutex> g(background_tasks_mutex);
	background_tasks.push_back(task);
	if(!background_thread)
	{
		background_thread = new std::thread(background_thread_proc);
	}
}

static void CryptonightR_build_program(
	std::vector<char>& ptx,
	std::string& lowered_name,
	const xmrstak_algo& algo,
	uint64_t height,
	uint32_t precompile_count,
	int arch_major,
	int arch_minor,
	std::string source)
{
	{
		CryptonightR_cache_mutex.WriteLock();

		// Remove old programs from cache
		for(size_t i = 0; i < CryptonightR_cache.size();)
		{
			const CacheEntry& entry = CryptonightR_cache[i];
			if((entry.algo == algo) && (entry.height + 2 + precompile_count < height))
			{
				printer::inst()->print_msg(LDEBUG, "CryptonightR: program for height %llu released (old program)", entry.height);
				CryptonightR_cache[i] = std::move(CryptonightR_cache.back());
				CryptonightR_cache.pop_back();
			}
			else
			{
				++i;
			}
		}
		CryptonightR_cache_mutex.UnLock();
	}

	ptx.clear();
	ptx.reserve(65536);

	std::lock_guard<std::mutex> g1(CryptonightR_build_mutex);
	{
		CryptonightR_cache_mutex.ReadLock();

		// Check if the cache already has this program (some other thread might have added it first)
		for(const CacheEntry& entry : CryptonightR_cache)
		{
			if((entry.algo == algo) && (entry.height == height) && (entry.arch_major == arch_major) && (entry.arch_minor == arch_minor))
			{
				ptx = entry.ptx;
				lowered_name = entry.lowered_name;
				CryptonightR_cache_mutex.UnLock();
				return;
			}
		}
		CryptonightR_cache_mutex.UnLock();
	}

	nvrtcProgram prog;
	nvrtcResult result = nvrtcCreateProgram(&prog, source.c_str(), "CryptonightR.curt", 0, NULL, NULL);
	if(result != NVRTC_SUCCESS)
	{
		printer::inst()->print_msg(L0, "nvrtcCreateProgram failed: %s", nvrtcGetErrorString(result));
		return;
	}

	result = nvrtcAddNameExpression(prog, "CryptonightR_phase2");
	if(result != NVRTC_SUCCESS)
	{
		printer::inst()->print_msg(L0, "nvrtcAddNameExpression failed: %s", nvrtcGetErrorString(result));
		nvrtcDestroyProgram(&prog);
		return;
	}

	char opt0[64];
	sprintf(opt0, "--gpu-architecture=compute_%d%d", arch_major, arch_minor);

	char opt1[64];
	sprintf(opt1, "-DALGO=%d", static_cast<int>(algo.Id()));

	const char* opts[2] = {opt0, opt1};

	result = nvrtcCompileProgram(prog, 2, opts);
	if(result != NVRTC_SUCCESS)
	{
		printer::inst()->print_msg(L0, "nvrtcCompileProgram failed: %s", nvrtcGetErrorString(result));

		size_t logSize;
		if(nvrtcGetProgramLogSize(prog, &logSize) == NVRTC_SUCCESS)
		{
			char* log = new char[logSize];
			if(nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS)
			{
				printer::inst()->print_msg(L0, "Program compile log: %s", log);
			}
			delete[] log;
		}
		nvrtcDestroyProgram(&prog);
		return;
	}

	const char* name;
	result = nvrtcGetLoweredName(prog, "CryptonightR_phase2", &name);
	if(result != NVRTC_SUCCESS)
	{
		printer::inst()->print_msg(L0, "nvrtcGetLoweredName failed: %s", nvrtcGetErrorString(result));
		nvrtcDestroyProgram(&prog);
		return;
	}

	size_t ptxSize;
	result = nvrtcGetPTXSize(prog, &ptxSize);
	if(result != NVRTC_SUCCESS)
	{
		printer::inst()->print_msg(L0, "nvrtcGetPTXSize failed: %s", nvrtcGetErrorString(result));
		nvrtcDestroyProgram(&prog);
		return;
	}

	ptx.resize(ptxSize);
	result = nvrtcGetPTX(prog, ptx.data());
	if(result != NVRTC_SUCCESS)
	{
		printer::inst()->print_msg(L0, "nvrtcGetPTX failed: %s", nvrtcGetErrorString(result));
		nvrtcDestroyProgram(&prog);
		return;
	}

	lowered_name = name;

	nvrtcDestroyProgram(&prog);

	printer::inst()->print_msg(LDEBUG, "CryptonightR: program for height %llu compiled", height);

	CryptonightR_cache_mutex.WriteLock();
	CryptonightR_cache.emplace_back(algo, height, arch_major, arch_minor, ptx, lowered_name);
	CryptonightR_cache_mutex.UnLock();
}

void CryptonightR_get_program(std::vector<char>& ptx, std::string& lowered_name, const xmrstak_algo algo, uint64_t height, uint32_t precompile_count, int arch_major, int arch_minor, bool background)
{
	if(background)
	{
		background_exec([=]() { std::vector<char> tmp; std::string s; CryptonightR_get_program(tmp, s, algo, height, precompile_count, arch_major, arch_minor, false); });
		return;
	}

	ptx.clear();

	const char* source_code_template =
#include "nvcc_code/cuda_cryptonight_r.curt"
		;
	const char include_name[] = "XMRSTAK_INCLUDE_RANDOM_MATH";
	const char* offset = strstr(source_code_template, include_name);
	if(!offset)
	{
		printer::inst()->print_msg(L0, "CryptonightR_get_program: XMRSTAK_INCLUDE_RANDOM_MATH not found in cuda_cryptonight_r.curt");
		return;
	}

	V4_Instruction code[256];
	int code_size;
	switch(algo.Id())
	{
	case cryptonight_r_wow:
		code_size = v4_random_math_init<cryptonight_r_wow>(code, height);
		break;
	case cryptonight_r:
		code_size = v4_random_math_init<cryptonight_r>(code, height);
		break;
		printer::inst()->print_msg(LDEBUG, "CryptonightR_get_program: invalid algo %d", algo);
		return;
	}

	std::string source_code(source_code_template, offset);
	source_code.append(get_code(code, code_size));
	source_code.append(offset + sizeof(include_name) - 1);

	{
		CryptonightR_cache_mutex.ReadLock();

		// Check if the cache has this program
		for(const CacheEntry& entry : CryptonightR_cache)
		{
			if((entry.algo == algo) && (entry.height == height) && (entry.arch_major == arch_major) && (entry.arch_minor == arch_minor))
			{
				printer::inst()->print_msg(LDEBUG, "CryptonightR: program for height %llu found in cache", height);
				ptx = entry.ptx;
				lowered_name = entry.lowered_name;
				CryptonightR_cache_mutex.UnLock();
				return;
			}
		}
		CryptonightR_cache_mutex.UnLock();
	}

	CryptonightR_build_program(ptx, lowered_name, algo, height, precompile_count, arch_major, arch_minor, source_code);
}

} // namespace nvidia
} // namespace xmrstak
