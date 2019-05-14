#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "xmrstak/backend/amd/OclCryptonightR_gen.hpp"
#include "xmrstak/backend/cpu/crypto/variant4_random_math.h"
#include "xmrstak/cpputil/read_write_lock.h"
#include "xmrstak/misc/console.hpp"

#include <chrono>
#include <iostream>
#include <regex>
#include <thread>

namespace xmrstak
{
namespace amd
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
		case ROL:
			s << 'r' << a << "=rotate(r" << a << ((inst.opcode == ROR) ? ",ROT_BITS-r" : ",r") << b << ");";
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
	CacheEntry(xmrstak_algo algo, uint64_t height_offset, size_t deviceIdx, cl_program program) :
		algo(algo),
		height_offset(height_offset),
		deviceIdx(deviceIdx),
		program(program)
	{
	}

	xmrstak_algo algo;
	uint64_t height_offset;
	size_t deviceIdx;
	cl_program program;
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

static cl_program search_program(
	const GpuContext* ctx,
	xmrstak_algo algo,
	uint64_t height_offset,
	bool lock_cache = true)
{
	if(lock_cache)
		CryptonightR_cache_mutex.ReadLock();

	// Check if the cache has this program
	for(const CacheEntry& entry : CryptonightR_cache)
	{
		if((entry.algo == algo) && (entry.height_offset == height_offset) && (entry.deviceIdx == ctx->deviceIdx))
		{
			printer::inst()->print_msg(LDEBUG, "CryptonightR: program for height_offset %llu found in cache", height_offset);
			auto result = entry.program;
			if(lock_cache)
				CryptonightR_cache_mutex.UnLock();
			return result;
		}
	}
	if(lock_cache)
		CryptonightR_cache_mutex.UnLock();

	return nullptr;
}

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

static cl_program CryptonightR_build_program(
	const GpuContext* ctx,
	xmrstak_algo algo,
	uint64_t height_offset,
	uint64_t height_chunk_size,
	uint32_t precompile_count,
	std::string source_code,
	std::string options)
{
	std::vector<cl_program> old_programs;
	old_programs.reserve(32);
	{
		CryptonightR_cache_mutex.WriteLock();

		// Remove old programs from cache
		for(size_t i = 0; i < CryptonightR_cache.size();)
		{
			const CacheEntry& entry = CryptonightR_cache[i];
			if((entry.algo == algo) && (entry.height_offset + (2 + precompile_count) * height_chunk_size < height_offset))
			{
				printer::inst()->print_msg(LDEBUG, "CryptonightR: program for height_offset %llu released (old program)", entry.height_offset);
				old_programs.push_back(entry.program);
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

	for(cl_program p : old_programs)
	{
		clReleaseProgram(p);
	}

	std::lock_guard<std::mutex> g1(CryptonightR_build_mutex);

	cl_program program = search_program(ctx, algo, height_offset);

	if(program)
	{
		return program;
	}

	cl_int ret;
	const char* source = source_code.c_str();

	program = clCreateProgramWithSource(ctx->opencl_ctx, 1, (const char**)&source, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L0, "Error %s when calling clCreateProgramWithSource on the OpenCL miner code", err_to_str(ret));
		return program;
	}

	ret = clBuildProgram(program, 1, &ctx->DeviceID, options.c_str(), NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		size_t len;
		printer::inst()->print_msg(L0, "Error %s when calling clBuildProgram.", err_to_str(ret));

		if((ret = clGetProgramBuildInfo(program, ctx->DeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L0, "Error %s when calling clGetProgramBuildInfo for length of build log output.", err_to_str(ret));
			return program;
		}

		char* BuildLog = (char*)malloc(len + 1);
		BuildLog[0] = '\0';

		if((ret = clGetProgramBuildInfo(program, ctx->DeviceID, CL_PROGRAM_BUILD_LOG, len, BuildLog, NULL)) != CL_SUCCESS)
		{
			free(BuildLog);
			printer::inst()->print_msg(L0, "Error %s when calling clGetProgramBuildInfo for build log.", err_to_str(ret));
			return program;
		}

		printer::inst()->print_str("Build log:\n");
		std::cerr << BuildLog << std::endl;

		free(BuildLog);
		return program;
	}

	cl_build_status status;
	do
	{
		if((ret = clGetProgramBuildInfo(program, ctx->DeviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L0, "Error %s when calling clGetProgramBuildInfo for status of build.", err_to_str(ret));
			return program;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	} while(status == CL_BUILD_IN_PROGRESS);

	CryptonightR_cache_mutex.WriteLock();
	auto cached_program = search_program(ctx, algo, height_offset, false);

	if(cached_program)
	{
		printer::inst()->print_msg(LDEBUG, "CryptonightR: release already existing program %llu", height_offset);
		clReleaseProgram(program);
		program = cached_program;
	}
	else
	{
		CryptonightR_cache.emplace_back(algo, height_offset, ctx->deviceIdx, program);
		printer::inst()->print_msg(LDEBUG, "CryptonightR: cache compiled program for height_offset %llu", height_offset);
	}

	CryptonightR_cache_mutex.UnLock();
	return program;
}

cl_program CryptonightR_get_program(GpuContext* ctx, xmrstak_algo algo, uint64_t height_offset, uint64_t height_chunk_size, uint32_t precompile_count, bool background)
{
	if(background)
	{
		background_exec([=]() { CryptonightR_get_program(ctx, algo, height_offset, height_chunk_size, precompile_count, false); });
		return nullptr;
	}

	auto program = search_program(ctx, algo, height_offset);

	if(program != nullptr)
		return program;

	printer::inst()->print_msg(LDEBUG, "CryptonightR: create code for block %llu to %llu", height_offset, height_offset + height_chunk_size);

	const char* source_code_definitions =
#include "amd_gpu/opencl/cryptonight_r_def.rtcl"
#include "amd_gpu/opencl/wolf-aes.cl"
		;

	const char* source_code_template =
#include "amd_gpu/opencl/cryptonight_r.rtcl"
		;
	const char include_name[] = "XMRSTAK_INCLUDE_RANDOM_MATH";
	const char* offset = strstr(source_code_template, include_name);
	if(!offset)
	{
		printer::inst()->print_msg(LDEBUG, "CryptonightR_get_program: XMRSTAK_INCLUDE_RANDOM_MATH not found in cryptonight_r.cl", algo);
		return nullptr;
	}

	std::string source_code(source_code_definitions);

	for(uint64_t c = 0; c < height_chunk_size; ++c)
	{
		V4_Instruction code[256];
		int code_size;
		switch(algo.Id())
		{
		case cryptonight_r_wow:
			code_size = v4_random_math_init<cryptonight_r_wow>(code, height_offset + c);
			break;
		case cryptonight_r:
			code_size = v4_random_math_init<cryptonight_r>(code, height_offset + c);
			break;
		default:
			printer::inst()->print_msg(L0, "CryptonightR_get_program: invalid algo %d", algo);
			return nullptr;
		}

		std::string kernel_code(source_code_template, offset);
		kernel_code.append(get_code(code, code_size));
		kernel_code.append(offset + sizeof(include_name) - 1);

		std::string kernel_name = "cn1_cryptonight_r_" + std::to_string(height_offset + c);

		source_code += std::regex_replace(kernel_code, std::regex("cn1_cryptonight_r"), kernel_name);
	}

	// scratchpad size for the selected mining algorithm
	size_t hashMemSize = algo.Mem();
	int threadMemMask = algo.Mask();
	int hashIterations = algo.Iter();

	size_t mem_chunk_exp = 1u << ctx->memChunk;
	size_t strided_index = ctx->stridedIndex;
	/* Adjust the config settings to a valid combination
	 * this is required if the dev pool is mining monero
	 * but the user tuned there settings for another currency
	 */
	if(algo == cryptonight_r || algo == cryptonight_r_wow)
	{
		if(ctx->memChunk < 2)
			mem_chunk_exp = 1u << 2;
		if(strided_index == 1)
			strided_index = 0;
	}

	// if intensity is a multiple of worksize than comp mode is not needed
	int needCompMode = ctx->compMode && ctx->rawIntensity % ctx->workSize != 0 ? 1 : 0;

	std::string options;
	options += " -DITERATIONS=" + std::to_string(hashIterations);
	options += " -DMASK=" + std::to_string(threadMemMask) + "U";
	options += " -DWORKSIZE=" + std::to_string(ctx->workSize) + "U";
	options += " -DSTRIDED_INDEX=" + std::to_string(strided_index);
	options += " -DMEM_CHUNK_EXPONENT=" + std::to_string(mem_chunk_exp) + "U";
	options += " -DCOMP_MODE=" + std::to_string(needCompMode);
	options += " -DMEMORY=" + std::to_string(hashMemSize) + "LU";
	options += " -DALGO=" + std::to_string(algo.Id());
	options += " -DCN_UNROLL=" + std::to_string(ctx->unroll);

	if(algo == cryptonight_gpu)
		options += " -cl-fp32-correctly-rounded-divide-sqrt";

	program = search_program(ctx, algo, height_offset);

	if(program != nullptr)
		return program;

	return CryptonightR_build_program(ctx, algo, height_offset, precompile_count, height_chunk_size, source_code, options);
}

} // namespace amd
} // namespace xmrstak
