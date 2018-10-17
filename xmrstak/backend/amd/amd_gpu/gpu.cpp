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
  */

#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/picosha2/picosha2.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/version.hpp"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <regex>
#include <cassert>
#include <algorithm>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#endif



#ifdef _WIN32
#include <windows.h>
#include <Shlobj.h>

static inline void create_directory(std::string dirname)
{
    _mkdir(dirname.data());
}

static inline std::string get_home()
{
	char path[MAX_PATH + 1];
	// get folder "appdata\local"
	if (SHGetSpecialFolderPathA(HWND_DESKTOP, path, CSIDL_LOCAL_APPDATA, FALSE))
	{
		return path;
	}
	else
		return ".";
}

static inline void port_sleep(size_t sec)
{
	Sleep(sec * 1000);
}
#else
#include <unistd.h>
#include <pwd.h>

static inline void create_directory(std::string dirname)
{
	mkdir(dirname.data(), 0744);
}

static inline std::string get_home()
{
	const char *home = ".";

	if ((home = getenv("HOME")) == nullptr)
		home = getpwuid(getuid())->pw_dir;

	return home;
}

static inline void port_sleep(size_t sec)
{
	sleep(sec);
}
#endif // _WIN32

#if 0
static inline long long unsigned int int_port(size_t i)
{
	return i;
}
#endif

#include "gpu.hpp"

const char* err_to_str(cl_int ret)
{
	switch(ret)
	{
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#ifdef CL_VERSION_1_2
	case CL_COMPILE_PROGRAM_FAILURE:
		return "CL_COMPILE_PROGRAM_FAILURE";
	case CL_LINKER_NOT_AVAILABLE:
		return "CL_LINKER_NOT_AVAILABLE";
	case CL_LINK_PROGRAM_FAILURE:
		return "CL_LINK_PROGRAM_FAILURE";
	case CL_DEVICE_PARTITION_FAILED:
		return "CL_DEVICE_PARTITION_FAILED";
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
#ifdef CL_VERSION_1_2
	case CL_INVALID_IMAGE_DESCRIPTOR:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:
		return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:
		return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#if defined(CL_VERSION_2_0) && !defined(CONF_ENFORCE_OpenCL_1_2)
	case CL_INVALID_PIPE_SIZE:
		return "CL_INVALID_PIPE_SIZE";
	case CL_INVALID_DEVICE_QUEUE:
		return "CL_INVALID_DEVICE_QUEUE";
#endif
	default:
		return "UNKNOWN_ERROR";
	}
}

#if 0
void printer::inst()->print_msg(L1,const char* fmt, ...);
void printer::inst()->print_str(const char* str);
#endif

char* LoadTextFile(const char* filename)
{
	size_t flen;
	char* out;
	FILE* kernel = fopen(filename, "rb");

	if(kernel == NULL)
		return NULL;

	fseek(kernel, 0, SEEK_END);
	flen = ftell(kernel);
	fseek(kernel, 0, SEEK_SET);

	out = (char*)malloc(flen+1);
	size_t r = fread(out, flen, 1, kernel);
	fclose(kernel);

	if(r != 1)
	{
		free(out);
		return NULL;
	}

	out[flen] = '\0';
	return out;
}

size_t InitOpenCLGpu(cl_context opencl_ctx, GpuContext* ctx, const char* source_code)
{
	size_t MaximumWorkSize;
	cl_int ret;

	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &MaximumWorkSize, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when querying a device's max worksize using clGetDeviceInfo.", err_to_str(ret));
		return ERR_OCL_API;
	}

	/* Some kernel spawn 8 times more threads than the user is configuring.
	 * To give the user the correct maximum work size we divide the hardware specific max by 8.
	 */
	MaximumWorkSize /= 8;
	printer::inst()->print_msg(L1,"Device %lu work size %lu / %lu.", ctx->deviceIdx, ctx->workSize, MaximumWorkSize);
#if defined(CL_VERSION_2_0) && !defined(CONF_ENFORCE_OpenCL_1_2)
	const cl_queue_properties CommandQueueProperties[] = { 0, 0, 0 };
	ctx->CommandQueues = clCreateCommandQueueWithProperties(opencl_ctx, ctx->DeviceID, CommandQueueProperties, &ret);
#else
	const cl_command_queue_properties CommandQueueProperties = { 0 };
	ctx->CommandQueues = clCreateCommandQueue(opencl_ctx, ctx->DeviceID, CommandQueueProperties, &ret);
#endif

	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateCommandQueueWithProperties.", err_to_str(ret));
		return ERR_OCL_API;
	}

	ctx->InputBuffer = clCreateBuffer(opencl_ctx, CL_MEM_READ_ONLY, 88, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create input buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	size_t scratchPadSize = std::max(
		cn_select_memory(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo()),
		cn_select_memory(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot())
	);

	size_t g_thd = ctx->rawIntensity;
	ctx->ExtraBuffers[0] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, scratchPadSize * g_thd, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create hash scratchpads buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	ctx->ExtraBuffers[1] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 200 * g_thd, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create hash states buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Blake-256 branches
	ctx->ExtraBuffers[2] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create Branch 0 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Groestl-256 branches
	ctx->ExtraBuffers[3] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create Branch 1 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// JH-256 branches
	ctx->ExtraBuffers[4] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create Branch 2 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Skein-512 branches
	ctx->ExtraBuffers[5] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create Branch 3 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Assume we may find up to 0xFF nonces in one run - it's reasonable
	ctx->OutputBuffer = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 0x100, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateBuffer to create output buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	std::vector<char> devNameVec(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DEVICE_NAME, devNameVec.size(), devNameVec.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_NAME for device %u.", err_to_str(ret),ctx->deviceIdx );
		return ERR_OCL_API;
	}

	std::vector<char> openCLDriverVer(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DRIVER_VERSION, openCLDriverVer.size(), openCLDriverVer.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get CL_DRIVER_VERSION for device %u.", err_to_str(ret),ctx->deviceIdx );
		return ERR_OCL_API;
	}

	xmrstak_algo miner_algo[2] = {
		::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo(),
		::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot()
	};
	int num_algos = miner_algo[0] == miner_algo[1] ? 1 : 2;

	for(int ii = 0; ii < num_algos; ++ii)
	{
		// scratchpad size for the selected mining algorithm
		size_t hashMemSize = cn_select_memory(miner_algo[ii]);
		int threadMemMask = cn_select_mask(miner_algo[ii]);
		int hashIterations = cn_select_iter(miner_algo[ii]);

		size_t mem_chunk_exp = 1u << ctx->memChunk;
		size_t strided_index = ctx->stridedIndex;
		/* Adjust the config settings to a valid combination
		 * this is required if the dev pool is mining monero
		 * but the user tuned there settings for another currency
		 */
		if(miner_algo[ii] == cryptonight_monero_v8)
		{
			if(ctx->memChunk < 2)
				mem_chunk_exp = 1u << 2;
			if(strided_index == 1)
				strided_index = 0;
		}

		std::string options;
		options += " -DITERATIONS=" + std::to_string(hashIterations);
		options += " -DMASK=" + std::to_string(threadMemMask);
		options += " -DWORKSIZE=" + std::to_string(ctx->workSize);
		options += " -DSTRIDED_INDEX=" + std::to_string(strided_index);
		options += " -DMEM_CHUNK_EXPONENT=" + std::to_string(mem_chunk_exp);
		options += " -DCOMP_MODE=" + std::to_string(ctx->compMode ? 1u : 0u);
		options += " -DMEMORY=" + std::to_string(hashMemSize);
		options += " -DALGO=" + std::to_string(miner_algo[ii]);
		options += " -DCN_UNROLL=" + std::to_string(ctx->unroll);
		/* AMD driver output is something like: `1445.5 (VM)`
		 * and is mapped to `14` only. The value is only used for a compiler
		 * workaround.
		 */
		options += " -DOPENCL_DRIVER_MAJOR=" + std::to_string(std::stoi(openCLDriverVer.data()) / 100);

		/* create a hash for the compile time cache
		 * used data:
		 *   - source code
		 *   - device name
		 *   - compile parameter
		 */
		std::string src_str(source_code);
		src_str += options;
		src_str += devNameVec.data();
		src_str += get_version_str();
		src_str += openCLDriverVer.data();

		std::string hash_hex_str;
		picosha2::hash256_hex_string(src_str, hash_hex_str);

		std::string cache_file = get_home() + "/.openclcache/" + hash_hex_str + ".openclbin";
		std::ifstream clBinFile(cache_file, std::ofstream::in | std::ofstream::binary);
		if(xmrstak::params::inst().AMDCache == false || !clBinFile.good())
		{
			if(xmrstak::params::inst().AMDCache)
				printer::inst()->print_msg(L1,"OpenCL device %u - Precompiled code %s not found. Compiling ...",ctx->deviceIdx, cache_file.c_str());
			ctx->Program[ii] = clCreateProgramWithSource(opencl_ctx, 1, (const char**)&source_code, NULL, &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clCreateProgramWithSource on the OpenCL miner code", err_to_str(ret));
				return ERR_OCL_API;
			}

			ret = clBuildProgram(ctx->Program[ii], 1, &ctx->DeviceID, options.c_str(), NULL, NULL);
			if(ret != CL_SUCCESS)
			{
				size_t len;
				printer::inst()->print_msg(L1,"Error %s when calling clBuildProgram.", err_to_str(ret));

				if((ret = clGetProgramBuildInfo(ctx->Program[ii], ctx->DeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1,"Error %s when calling clGetProgramBuildInfo for length of build log output.", err_to_str(ret));
					return ERR_OCL_API;
				}

				char* BuildLog = (char*)malloc(len + 1);
				BuildLog[0] = '\0';

				if((ret = clGetProgramBuildInfo(ctx->Program[ii], ctx->DeviceID, CL_PROGRAM_BUILD_LOG, len, BuildLog, NULL)) != CL_SUCCESS)
				{
					free(BuildLog);
					printer::inst()->print_msg(L1,"Error %s when calling clGetProgramBuildInfo for build log.", err_to_str(ret));
					return ERR_OCL_API;
				}

				printer::inst()->print_str("Build log:\n");
				std::cerr<<BuildLog<<std::endl;

				free(BuildLog);
				return ERR_OCL_API;
			}

			cl_uint num_devices;
			clGetProgramInfo(ctx->Program[ii], CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices,NULL);


			std::vector<cl_device_id> devices_ids(num_devices);
			clGetProgramInfo(ctx->Program[ii], CL_PROGRAM_DEVICES, sizeof(cl_device_id)* devices_ids.size(), devices_ids.data(),NULL);
			int dev_id = 0;
			/* Search for the gpu within the program context.
			 * The id can be different to  ctx->DeviceID.
			 */
			for(auto & ocl_device : devices_ids)
			{
				if(ocl_device == ctx->DeviceID)
					break;
				dev_id++;
			}

			cl_build_status status;
			do
			{
				if((ret = clGetProgramBuildInfo(ctx->Program[ii], ctx->DeviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1,"Error %s when calling clGetProgramBuildInfo for status of build.", err_to_str(ret));
					return ERR_OCL_API;
				}
				port_sleep(1);
			}
			while(status == CL_BUILD_IN_PROGRESS);

			if(xmrstak::params::inst().AMDCache)
			{
				std::vector<size_t> binary_sizes(num_devices);
				clGetProgramInfo (ctx->Program[ii], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * binary_sizes.size(), binary_sizes.data(), NULL);

				std::vector<char*> all_programs(num_devices);
				std::vector<std::vector<char>> program_storage;

				int p_id = 0;
				size_t mem_size = 0;
				// create memory  structure to query all OpenCL program binaries
				for(auto & p : all_programs)
				{
					program_storage.emplace_back(std::vector<char>(binary_sizes[p_id]));
					all_programs[p_id] = program_storage[p_id].data();
					mem_size += binary_sizes[p_id];
					p_id++;
				}

				if((ret = clGetProgramInfo(ctx->Program[ii], CL_PROGRAM_BINARIES, num_devices * sizeof(char*), all_programs.data(),NULL)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1,"Error %s when calling clGetProgramInfo.", err_to_str(ret));
					return ERR_OCL_API;
				}

				std::ofstream file_stream;
				file_stream.open(cache_file, std::ofstream::out | std::ofstream::binary);
				file_stream.write(all_programs[dev_id], binary_sizes[dev_id]);
				file_stream.close();
				printer::inst()->print_msg(L1, "OpenCL device %u - Precompiled code stored in file %s",ctx->deviceIdx, cache_file.c_str());
			}
		}
		else
		{
			printer::inst()->print_msg(L1, "OpenCL device %u - Load precompiled code from file %s",ctx->deviceIdx, cache_file.c_str());
			std::ostringstream ss;
			ss << clBinFile.rdbuf();
			std::string s = ss.str();

			size_t bin_size = s.size();
			auto data_ptr = s.data();

			cl_int clStatus;
			ctx->Program[ii] = clCreateProgramWithBinary(
				opencl_ctx, 1, &ctx->DeviceID, &bin_size,
				(const unsigned char **)&data_ptr, &clStatus, &ret
			);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clCreateProgramWithBinary. Try to delete file %s", err_to_str(ret), cache_file.c_str());
				return ERR_OCL_API;
			}
			ret = clBuildProgram(ctx->Program[ii], 1, &ctx->DeviceID, NULL, NULL, NULL);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clBuildProgram. Try to delete file %s", err_to_str(ret), cache_file.c_str());
				return ERR_OCL_API;
			}
		}

		std::vector<std::string> KernelNames = { "cn0", "cn1", "cn2", "Blake", "Groestl", "JH", "Skein" };
		// append algorithm number to kernel name
		for(int k = 0; k < 3; k++)
			KernelNames[k] += std::to_string(miner_algo[ii]);

		if(ii == 0)
		{
			for(int i = 0; i < 7; ++i)
			{
				ctx->Kernels[ii][i] = clCreateKernel(ctx->Program[ii], KernelNames[i].c_str(), &ret);
				if(ret != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1,"Error %s when calling clCreateKernel for kernel_0 %s.", err_to_str(ret), KernelNames[i].c_str());
					return ERR_OCL_API;
				}
			}
		}
		else
		{
			for(int i = 0; i < 3; ++i)
			{
				ctx->Kernels[ii][i] = clCreateKernel(ctx->Program[ii], KernelNames[i].c_str(), &ret);
				if(ret != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1,"Error %s when calling clCreateKernel for kernel_1 %s.", err_to_str(ret), KernelNames[i].c_str());
					return ERR_OCL_API;
				}
			}
			// move kernel from the main algorithm into the root algorithm kernel space
			for(int i = 3; i < 7; ++i)
			{
				ctx->Kernels[ii][i] = ctx->Kernels[0][i];
			}

		}
	}
	ctx->Nonce = 0;
	return 0;
}

const cl_platform_info attributeTypes[5] = {
	CL_PLATFORM_NAME,
	CL_PLATFORM_VENDOR,
	CL_PLATFORM_VERSION,
	CL_PLATFORM_PROFILE,
	CL_PLATFORM_EXTENSIONS
};

const char* const attributeNames[] = {
	"CL_PLATFORM_NAME",
	"CL_PLATFORM_VENDOR",
	"CL_PLATFORM_VERSION",
	"CL_PLATFORM_PROFILE",
	"CL_PLATFORM_EXTENSIONS"
};

#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

uint32_t getNumPlatforms()
{
	cl_uint num_platforms = 0;
	cl_platform_id * platforms = NULL;
	cl_int clStatus;

	// Get platform and device information
	clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	if(clStatus != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetPlatformIDs for number of platforms.", err_to_str(clStatus));
		return 0u;
	}

	return num_platforms;
}

std::vector<GpuContext> getAMDDevices(int index)
{
	std::vector<GpuContext> ctxVec;
	std::vector<cl_platform_id> platforms;
	std::vector<cl_device_id> device_list;

	cl_int clStatus;
	cl_uint num_devices;
	uint32_t numPlatforms = getNumPlatforms();

	if(numPlatforms == 0)
		return ctxVec;

	platforms.resize(numPlatforms);
	if((clStatus = clGetPlatformIDs(numPlatforms, platforms.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetPlatformIDs for platform information.", err_to_str(clStatus));
		return ctxVec;
	}

	if((clStatus = clGetDeviceIDs( platforms[index], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceIDs for of devices.", err_to_str(clStatus));
		return ctxVec;
	}

	device_list.resize(num_devices);
	if((clStatus = clGetDeviceIDs( platforms[index], CL_DEVICE_TYPE_GPU, num_devices, device_list.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceIDs for device information.", err_to_str(clStatus));
		return ctxVec;
	}

	for (size_t k = 0; k < num_devices; k++)
	{
		std::vector<char> devVendorVec(1024);
		if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_VENDOR, devVendorVec.size(), devVendorVec.data(), NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get the device vendor name for device %u.", err_to_str(clStatus), k);
			continue;
		}

		std::string devVendor(devVendorVec.data());

		bool isAMDDevice = devVendor.find("Advanced Micro Devices") != std::string::npos || devVendor.find("AMD") != std::string::npos;
		bool isNVIDIADevice = devVendor.find("NVIDIA Corporation") != std::string::npos || devVendor.find("NVIDIA") != std::string::npos;

		std::string selectedOpenCLVendor = xmrstak::params::inst().openCLVendor;
		if((isAMDDevice && selectedOpenCLVendor == "AMD") || (isNVIDIADevice && selectedOpenCLVendor == "NVIDIA"))
		{
			GpuContext ctx;
			std::vector<char> devNameVec(1024);
			size_t maxMem;
			if( devVendor.find("NVIDIA Corporation") != std::string::npos)
				ctx.isNVIDIA = true;

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &(ctx.computeUnits), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_MAX_COMPUTE_UNITS for device %u.", err_to_str(clStatus), k);
				continue;
			}

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &(maxMem), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_MAX_MEM_ALLOC_SIZE for device %u.", err_to_str(clStatus), k);
				continue;
			}

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &(ctx.freeMem), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_GLOBAL_MEM_SIZE for device %u.", err_to_str(clStatus), k);
				continue;
			}

			// the allocation for NVIDIA OpenCL is not limited to 1/4 of the GPU memory per allocation
			if(ctx.isNVIDIA)
				maxMem = ctx.freeMem;

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_NAME, devNameVec.size(), devNameVec.data(), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_NAME for device %u.", err_to_str(clStatus), k);
				continue;
			}

			// if environment variable GPU_SINGLE_ALLOC_PERCENT is not set we can not allocate the full memory
			ctx.deviceIdx = k;
			ctx.freeMem = std::min(ctx.freeMem, maxMem);
			ctx.name = std::string(devNameVec.data());
			ctx.DeviceID = device_list[k];
			printer::inst()->print_msg(L0,"Found OpenCL GPU %s.",ctx.name.c_str());
			ctxVec.push_back(ctx);
		}
	}

	return ctxVec;
}

int getAMDPlatformIdx()
{

	uint32_t numPlatforms = getNumPlatforms();

	if(numPlatforms == 0)
	{
		printer::inst()->print_msg(L0,"WARNING: No OpenCL platform found.");
		return -1;
	}
	cl_platform_id * platforms = NULL;
	cl_int clStatus;

	platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numPlatforms);
	clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);

	int platformIndex = -1;
	// Mesa OpenCL is the fallback if no AMD or Apple OpenCL is found
	int mesaPlatform = -1;

	if(clStatus == CL_SUCCESS)
	{
		for (int i = 0; i < numPlatforms; i++) {
			size_t infoSize;
			clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &infoSize);
			std::vector<char> platformNameVec(infoSize);

			clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, infoSize, platformNameVec.data(), NULL);
			std::string platformName(platformNameVec.data());

			bool isAMDOpenCL = platformName.find("Advanced Micro Devices") != std::string::npos ||
				platformName.find("Apple") != std::string::npos ||
				platformName.find("Mesa") != std::string::npos;
			bool isNVIDIADevice = platformName.find("NVIDIA Corporation") != std::string::npos || platformName.find("NVIDIA") != std::string::npos;
			std::string selectedOpenCLVendor = xmrstak::params::inst().openCLVendor;
			if((isAMDOpenCL && selectedOpenCLVendor == "AMD") || (isNVIDIADevice && selectedOpenCLVendor == "NVIDIA"))
			{
				printer::inst()->print_msg(L0,"Found %s platform index id = %i, name = %s", selectedOpenCLVendor.c_str(), i , platformName.c_str());
				if(platformName.find("Mesa") != std::string::npos)
					mesaPlatform = i;
				else
				{
					// exit if AMD or Apple platform is found
					platformIndex = i;
					break;
				}
			}
		}
		// fall back to Mesa OpenCL
		if(platformIndex == -1 && mesaPlatform != -1)
		{
			printer::inst()->print_msg(L0,"No AMD platform found select Mesa as OpenCL platform");
			platformIndex = mesaPlatform;
		}
	}
	else
		printer::inst()->print_msg(L1,"WARNING: %s when calling clGetPlatformIDs for platform information.", err_to_str(clStatus));

	free(platforms);
	return platformIndex;
}

// RequestedDeviceIdxs is a list of OpenCL device indexes
// NumDevicesRequested is number of devices in RequestedDeviceIdxs list
// Returns 0 on success, -1 on stupid params, -2 on OpenCL API error
size_t InitOpenCL(GpuContext* ctx, size_t num_gpus, size_t platform_idx)
{

	cl_context opencl_ctx;
	cl_int ret;
	cl_uint entries;

	if((ret = clGetPlatformIDs(0, NULL, &entries)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clGetPlatformIDs for number of platforms.", err_to_str(ret));
		return ERR_OCL_API;
	}


	// The number of platforms naturally is the index of the last platform plus one.
	if(entries <= platform_idx)
	{
		printer::inst()->print_msg(L1,"Selected OpenCL platform index %d doesn't exist.", platform_idx);
		return ERR_STUPID_PARAMS;
	}

	/*MSVC skimping on devel costs by shoehorning C99 to be a subset of C++? Noooo... can't be.*/
#ifdef __GNUC__
	cl_platform_id PlatformIDList[entries];
#else
	cl_platform_id* PlatformIDList = (cl_platform_id*)_alloca(entries * sizeof(cl_platform_id));
#endif
	if((ret = clGetPlatformIDs(entries, PlatformIDList, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clGetPlatformIDs for platform ID information.", err_to_str(ret));
		return ERR_OCL_API;
	}

	size_t infoSize;
	clGetPlatformInfo(PlatformIDList[platform_idx], CL_PLATFORM_VENDOR, 0, NULL, &infoSize);
	std::vector<char> platformNameVec(infoSize);
	clGetPlatformInfo(PlatformIDList[platform_idx], CL_PLATFORM_VENDOR, infoSize, platformNameVec.data(), NULL);
	std::string platformName(platformNameVec.data());
	if(xmrstak::params::inst().openCLVendor == "AMD" && platformName.find("Advanced Micro Devices") == std::string::npos)
	{
		printer::inst()->print_msg(L1,"WARNING: using non AMD device: %s", platformName.c_str());
	}

	if((ret = clGetDeviceIDs(PlatformIDList[platform_idx], CL_DEVICE_TYPE_GPU, 0, NULL, &entries)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clGetDeviceIDs for number of devices.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Same as the platform index sanity check, except we must check all requested device indexes
	for(int i = 0; i < num_gpus; ++i)
	{
		if(entries <= ctx[i].deviceIdx)
		{
			printer::inst()->print_msg(L1,"Selected OpenCL device index %lu doesn't exist.\n", ctx[i].deviceIdx);
			return ERR_STUPID_PARAMS;
		}
	}

#ifdef __GNUC__
	cl_device_id DeviceIDList[entries];
#else
	cl_device_id* DeviceIDList = (cl_device_id*)_alloca(entries * sizeof(cl_device_id));
#endif
	if((ret = clGetDeviceIDs(PlatformIDList[platform_idx], CL_DEVICE_TYPE_GPU, entries, DeviceIDList, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clGetDeviceIDs for device ID information.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Indexes sanity checked above
#ifdef __GNUC__
	cl_device_id TempDeviceList[num_gpus];
#else
	cl_device_id* TempDeviceList = (cl_device_id*)_alloca(entries * sizeof(cl_device_id));
#endif
	for(int i = 0; i < num_gpus; ++i)
	{
		ctx[i].DeviceID = DeviceIDList[ctx[i].deviceIdx];
		TempDeviceList[i] = DeviceIDList[ctx[i].deviceIdx];
	}

	opencl_ctx = clCreateContext(NULL, num_gpus, TempDeviceList, NULL, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateContext.", err_to_str(ret));
		return ERR_OCL_API;
	}

	//char* source_code = LoadTextFile(sSourcePath);

	const char *fastIntMathV2CL =
			#include "./opencl/fast_int_math_v2.cl"
	;
	const char *cryptonightCL =
			#include "./opencl/cryptonight.cl"
	;
	const char *blake256CL =
			#include "./opencl/blake256.cl"
	;
	const char *groestl256CL =
			#include "./opencl/groestl256.cl"
	;
	const char *jhCL =
			#include "./opencl/jh.cl"
	;
	const char *wolfAesCL =
			#include "./opencl/wolf-aes.cl"
	;
	const char *wolfSkeinCL =
			#include "./opencl/wolf-skein.cl"
	;

	std::string source_code(cryptonightCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_FAST_INT_MATH_V2"), fastIntMathV2CL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_WOLF_AES"), wolfAesCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_WOLF_SKEIN"), wolfSkeinCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_JH"), jhCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_BLAKE256"), blake256CL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_GROESTL256"), groestl256CL);

	// create a directory  for the OpenCL compile cache
	create_directory(get_home() + "/.openclcache");

	for(int i = 0; i < num_gpus; ++i)
	{
		const std::string backendName = xmrstak::params::inst().openCLVendor;
		if(ctx[i].stridedIndex == 2 && (ctx[i].rawIntensity % ctx[i].workSize) != 0)
		{
			size_t reduced_intensity = (ctx[i].rawIntensity / ctx[i].workSize) * ctx[i].workSize;
			ctx[i].rawIntensity = reduced_intensity;
			printer::inst()->print_msg(L0, "WARNING %s: gpu %d intensity is not a multiple of 'worksize', auto reduce intensity to %d", backendName.c_str(), ctx[i].deviceIdx, int(reduced_intensity));
		}

		if((ret = InitOpenCLGpu(opencl_ctx, &ctx[i], source_code.c_str())) != ERR_SUCCESS)
		{
			return ret;
		}
	}

	return ERR_SUCCESS;
}

size_t XMRSetJob(GpuContext* ctx, uint8_t* input, size_t input_len, uint64_t target, xmrstak_algo miner_algo)
{
	// switch to the kernel storage
	int kernel_storage = miner_algo == ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() ? 0 : 1;

	cl_int ret;

	if(input_len > 84)
		return ERR_STUPID_PARAMS;

	input[input_len] = 0x01;
	memset(input + input_len + 1, 0, 88 - input_len - 1);

	size_t numThreads = ctx->rawIntensity;

	if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->InputBuffer, CL_TRUE, 0, 88, input, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueWriteBuffer to fill input buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][0], 0, sizeof(cl_mem), &ctx->InputBuffer)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Scratchpads
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][0], 1, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][0], 2, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 2.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][0], 3, sizeof(cl_ulong), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 3.", err_to_str(ret));
		return(ERR_OCL_API);
	}

	// CN1 Kernel

	// Scratchpads
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][1], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 1, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][1], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 1, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][1], 2, sizeof(cl_ulong), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 1, argument 2.", err_to_str(ret));
		return(ERR_OCL_API);
	}

	if(miner_algo == cryptonight_monero || miner_algo == cryptonight_aeon || miner_algo == cryptonight_ipbc || miner_algo == cryptonight_stellite || miner_algo == cryptonight_masari || miner_algo == cryptonight_bittube2)
	{
		// Input
		if ((ret = clSetKernelArg(ctx->Kernels[kernel_storage][1], 3, sizeof(cl_mem), &ctx->InputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 1, argument 4(input buffer).", err_to_str(ret));
			return ERR_OCL_API;
		}
	}

	// CN3 Kernel
	// Scratchpads
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 0
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 2, sizeof(cl_mem), ctx->ExtraBuffers + 2)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 2.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 1
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 3, sizeof(cl_mem), ctx->ExtraBuffers + 3)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 3.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 2
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 4, sizeof(cl_mem), ctx->ExtraBuffers + 4)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 4.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 3
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 5, sizeof(cl_mem), ctx->ExtraBuffers + 5)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 5.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][2], 6, sizeof(cl_ulong), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 6.", err_to_str(ret));
		return(ERR_OCL_API);
	}

	for(int i = 0; i < 4; ++i)
	{
		// States
		if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][i + 3], 0, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 0);
			return ERR_OCL_API;
		}

		// Nonce buffer
		if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][i + 3], 1, sizeof(cl_mem), ctx->ExtraBuffers + (i + 2))) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 1);
			return ERR_OCL_API;
		}

		// Output
		if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][i + 3], 2, sizeof(cl_mem), &ctx->OutputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 2);
			return ERR_OCL_API;
		}

		// Target
		if((ret = clSetKernelArg(ctx->Kernels[kernel_storage][i + 3], 3, sizeof(cl_ulong), &target)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 3);
			return ERR_OCL_API;
		}
	}

	return ERR_SUCCESS;
}

size_t XMRRunJob(GpuContext* ctx, cl_uint* HashOutput, xmrstak_algo miner_algo)
{
	// switch to the kernel storage
	int kernel_storage = miner_algo == ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() ? 0 : 1;

	cl_int ret;
	cl_uint zero = 0;
	size_t BranchNonces[4];
	memset(BranchNonces,0,sizeof(size_t)*4);

	size_t g_intensity = ctx->rawIntensity;
	size_t w_size = ctx->workSize;
	size_t g_thd = g_intensity;

	if(ctx->compMode)
	{
		// round up to next multiple of w_size
		g_thd = ((g_intensity + w_size - 1u) / w_size) * w_size;
		// number of global threads must be a multiple of the work group size (w_size)
		assert(g_thd%w_size == 0);
	}

	for(int i = 2; i < 6; ++i)
	{
		if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->ExtraBuffers[i], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), &zero, 0, NULL, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clEnqueueWriteBuffer to zero branch buffer counter %d.", err_to_str(ret), i - 2);
			return ERR_OCL_API;
		}
	}

	if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->OutputBuffer, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(cl_uint), &zero, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	clFinish(ctx->CommandQueues);

	size_t Nonce[2] = {ctx->Nonce, 1}, gthreads[2] = { g_thd, 8 }, lthreads[2] = { w_size, 8 };
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[kernel_storage][0], 2, Nonce, gthreads, lthreads, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 0);
		return ERR_OCL_API;
	}

	size_t tmpNonce = ctx->Nonce;

	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[kernel_storage][1], 1, &tmpNonce, &g_thd, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 1);
		return ERR_OCL_API;
	}

	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[kernel_storage][2], 2, Nonce, gthreads, lthreads, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 2);
		return ERR_OCL_API;
	}

	if((ret = clEnqueueReadBuffer(ctx->CommandQueues, ctx->ExtraBuffers[2], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), BranchNonces, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if((ret = clEnqueueReadBuffer(ctx->CommandQueues, ctx->ExtraBuffers[3], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), BranchNonces + 1, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if((ret = clEnqueueReadBuffer(ctx->CommandQueues, ctx->ExtraBuffers[4], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), BranchNonces + 2, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if((ret = clEnqueueReadBuffer(ctx->CommandQueues, ctx->ExtraBuffers[5], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), BranchNonces + 3, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	clFinish(ctx->CommandQueues);

	for(int i = 0; i < 4; ++i)
	{
		if(BranchNonces[i])
		{
			// Threads
			if((clSetKernelArg(ctx->Kernels[kernel_storage][i + 3], 4, sizeof(cl_ulong), BranchNonces + i)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 4);
				return(ERR_OCL_API);
			}

			// round up to next multiple of w_size
			BranchNonces[i] = ((BranchNonces[i] + w_size - 1u) / w_size) * w_size;
			// number of global threads must be a multiple of the work group size (w_size)
			assert(BranchNonces[i]%w_size == 0);
			size_t tmpNonce = ctx->Nonce;
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[kernel_storage][i + 3], 1, &tmpNonce, BranchNonces + i, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), i + 3);
				return ERR_OCL_API;
			}
		}
	}

	if((ret = clEnqueueReadBuffer(ctx->CommandQueues, ctx->OutputBuffer, CL_TRUE, 0, sizeof(cl_uint) * 0x100, HashOutput, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	clFinish(ctx->CommandQueues);
	auto & numHashValues = HashOutput[0xFF];
	// avoid out of memory read, we have only storage for 0xFF results
	if(numHashValues > 0xFF)
		numHashValues = 0xFF;
	ctx->Nonce += g_intensity;

	return ERR_SUCCESS;
}
