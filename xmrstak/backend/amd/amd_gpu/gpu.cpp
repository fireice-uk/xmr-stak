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

#include "xmrstak/backend/amd/OclCryptonightR_gen.hpp"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/net/msgstruct.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/picosha2/picosha2.hpp"
#include "xmrstak/version.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <math.h>
#include <regex>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifdef _WIN32
#include <windows.h>

static inline void create_directory(std::string dirname)
{
	_mkdir(dirname.data());
}

static inline void port_sleep(size_t sec)
{
	Sleep(sec * 1000);
}
#else
#include <pwd.h>
#include <unistd.h>

static inline void create_directory(std::string dirname)
{
	mkdir(dirname.data(), 0744);
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

	out = (char*)malloc(flen + 1);
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
		printer::inst()->print_msg(L1, "Error %s when querying a device's max worksize using clGetDeviceInfo.", err_to_str(ret));
		return ERR_OCL_API;
	}

	auto neededAlgorithms = ::jconf::inst()->GetCurrentCoinSelection().GetAllAlgorithms();
	bool useCryptonight_gpu = std::find(neededAlgorithms.begin(), neededAlgorithms.end(), cryptonight_gpu) != neededAlgorithms.end();

	if(useCryptonight_gpu)
	{
		// work cn_1 we use 16x more threads than configured by the user
		MaximumWorkSize /= 16;
	}
	else
	{
		/* Some kernel spawn 8 times more threads than the user is configuring.
		 * To give the user the correct maximum work size we divide the hardware specific max by 8.
		 */
		MaximumWorkSize /= 8;
	}
	printer::inst()->print_msg(L1, "Device %lu work size %lu / %lu.", ctx->deviceIdx, ctx->workSize, MaximumWorkSize);

	if(ctx->workSize > MaximumWorkSize)
	{
		ctx->workSize = MaximumWorkSize;
		printer::inst()->print_msg(L1, "Device %lu work size to large, reduce to %lu / %lu.", ctx->deviceIdx, ctx->workSize, MaximumWorkSize);
	}

	const std::string backendName = xmrstak::params::inst().openCLVendor;
	if((ctx->stridedIndex == 2 || ctx->stridedIndex == 3) && (ctx->rawIntensity % ctx->workSize) != 0)
	{
		size_t reduced_intensity = (ctx->rawIntensity / ctx->workSize) * ctx->workSize;
		ctx->rawIntensity = reduced_intensity;
		printer::inst()->print_msg(L0, "WARNING %s: gpu %d intensity is not a multiple of 'worksize', auto reduce intensity to %d", backendName.c_str(), ctx->deviceIdx, int(reduced_intensity));
	}

#if defined(CL_VERSION_2_0) && !defined(CONF_ENFORCE_OpenCL_1_2)
	const cl_queue_properties CommandQueueProperties[] = {0, 0, 0};
	ctx->CommandQueues = clCreateCommandQueueWithProperties(opencl_ctx, ctx->DeviceID, CommandQueueProperties, &ret);
#else
	const cl_command_queue_properties CommandQueueProperties = {0};
	ctx->CommandQueues = clCreateCommandQueue(opencl_ctx, ctx->DeviceID, CommandQueueProperties, &ret);
#endif

	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateCommandQueueWithProperties.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &(ctx->computeUnits), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_MAX_COMPUTE_UNITS for device %u.", err_to_str(ret), (uint32_t)ctx->deviceIdx);
		return ERR_OCL_API;
	}

	ctx->InputBuffer = clCreateBuffer(opencl_ctx, CL_MEM_READ_ONLY, 128, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create input buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	size_t scratchPadSize = 0;
	for(const auto algo : neededAlgorithms)
	{
		scratchPadSize = std::max(scratchPadSize, algo.Mem());
	}

	size_t g_thd = ctx->rawIntensity;
	ctx->ExtraBuffers[0] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, scratchPadSize * g_thd, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create hash scratchpads buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	ctx->ExtraBuffers[1] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 200 * g_thd, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create hash states buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Blake-256 branches
	ctx->ExtraBuffers[2] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create Branch 0 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Groestl-256 branches
	ctx->ExtraBuffers[3] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create Branch 1 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// JH-256 branches
	ctx->ExtraBuffers[4] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create Branch 2 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Skein-512 branches
	ctx->ExtraBuffers[5] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create Branch 3 buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Assume we may find up to 0xFF nonces in one run - it's reasonable
	ctx->OutputBuffer = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 0x100, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create output buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	std::vector<char> devNameVec(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DEVICE_NAME, devNameVec.size(), devNameVec.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_NAME for device %u.", err_to_str(ret), ctx->deviceIdx);
		return ERR_OCL_API;
	}

	std::vector<char> openCLDriverVer(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DRIVER_VERSION, openCLDriverVer.size(), openCLDriverVer.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DRIVER_VERSION for device %u.", err_to_str(ret), ctx->deviceIdx);
		return ERR_OCL_API;
	}

	for(const auto miner_algo : neededAlgorithms)
	{
		// scratchpad size for the selected mining algorithm
		size_t hashMemSize = miner_algo.Mem();
		int threadMemMask = miner_algo.Mask();
		int hashIterations = miner_algo.Iter();

		size_t mem_chunk_exp = 1u << ctx->memChunk;
		size_t strided_index = ctx->stridedIndex;
		/* Adjust the config settings to a valid combination
		 * this is required if the dev pool is mining monero
		 * but the user tuned there settings for another currency
		 */
		if(miner_algo == cryptonight_monero_v8 || miner_algo == cryptonight_v8_reversewaltz)
		{
			if(ctx->memChunk < 2)
				mem_chunk_exp = 1u << 2;
			if(strided_index == 1)
				strided_index = 0;
		}

		if(miner_algo == cryptonight_gpu)
		{
			strided_index = 0;
		}

		if(miner_algo == cryptonight_r || miner_algo == cryptonight_r_wow)
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
		options += " -DALGO=" + std::to_string(miner_algo.Id());
		options += " -DCN_UNROLL=" + std::to_string(ctx->unroll);
		/* AMD driver output is something like: `1445.5 (VM)`
		 * and is mapped to `14` only. The value is only used for a compiler
		 * workaround.
		 */
		options += " -DOPENCL_DRIVER_MAJOR=" + std::to_string(std::stoi(openCLDriverVer.data()) / 100);

		uint32_t isWindowsOs = 0;
#ifdef _WIN32
		isWindowsOs = 1;
#endif
		options += " -DIS_WINDOWS_OS=" + std::to_string(isWindowsOs);

		if(miner_algo == cryptonight_gpu)
			options += " -cl-fp32-correctly-rounded-divide-sqrt";

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

		const std::string cache_dir = xmrstak::params::inst().rootAMDCacheDir;

		std::string cache_file = cache_dir + hash_hex_str + ".openclbin";
		std::ifstream clBinFile(cache_file, std::ofstream::in | std::ofstream::binary);
		if(xmrstak::params::inst().AMDCache == false || !clBinFile.good())
		{
			if(xmrstak::params::inst().AMDCache)
				printer::inst()->print_msg(L1, "OpenCL device %u - Precompiled code %s not found. Compiling ...", ctx->deviceIdx, cache_file.c_str());
			ctx->Program[miner_algo] = clCreateProgramWithSource(opencl_ctx, 1, (const char**)&source_code, NULL, &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clCreateProgramWithSource on the OpenCL miner code", err_to_str(ret));
				return ERR_OCL_API;
			}

			ret = clBuildProgram(ctx->Program[miner_algo], 1, &ctx->DeviceID, options.c_str(), NULL, NULL);
			if(ret != CL_SUCCESS)
			{
				size_t len;
				printer::inst()->print_msg(L1, "Error %s when calling clBuildProgram.", err_to_str(ret));

				if((ret = clGetProgramBuildInfo(ctx->Program[miner_algo], ctx->DeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1, "Error %s when calling clGetProgramBuildInfo for length of build log output.", err_to_str(ret));
					return ERR_OCL_API;
				}

				char* BuildLog = (char*)malloc(len + 1);
				BuildLog[0] = '\0';

				if((ret = clGetProgramBuildInfo(ctx->Program[miner_algo], ctx->DeviceID, CL_PROGRAM_BUILD_LOG, len, BuildLog, NULL)) != CL_SUCCESS)
				{
					free(BuildLog);
					printer::inst()->print_msg(L1, "Error %s when calling clGetProgramBuildInfo for build log.", err_to_str(ret));
					return ERR_OCL_API;
				}

				printer::inst()->print_str("Build log:\n");
				std::cerr << BuildLog << std::endl;

				free(BuildLog);
				return ERR_OCL_API;
			}

			cl_uint num_devices;
			clGetProgramInfo(ctx->Program[miner_algo], CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

			std::vector<cl_device_id> devices_ids(num_devices);
			clGetProgramInfo(ctx->Program[miner_algo], CL_PROGRAM_DEVICES, sizeof(cl_device_id) * devices_ids.size(), devices_ids.data(), NULL);
			int dev_id = 0;
			/* Search for the gpu within the program context.
			 * The id can be different to  ctx->DeviceID.
			 */
			for(auto& ocl_device : devices_ids)
			{
				if(ocl_device == ctx->DeviceID)
					break;
				dev_id++;
			}

			cl_build_status status;
			do
			{
				if((ret = clGetProgramBuildInfo(ctx->Program[miner_algo], ctx->DeviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1, "Error %s when calling clGetProgramBuildInfo for status of build.", err_to_str(ret));
					return ERR_OCL_API;
				}
				port_sleep(1);
			} while(status == CL_BUILD_IN_PROGRESS);

			if(xmrstak::params::inst().AMDCache)
			{
				std::vector<size_t> binary_sizes(num_devices);
				clGetProgramInfo(ctx->Program[miner_algo], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * binary_sizes.size(), binary_sizes.data(), NULL);

				std::vector<char*> all_programs(num_devices);
				std::vector<std::vector<char>> program_storage;

				int p_id = 0;
				size_t mem_size = 0;
				// create memory  structure to query all OpenCL program binaries
				for(auto& p : all_programs)
				{
					program_storage.emplace_back(std::vector<char>(binary_sizes[p_id]));
					all_programs[p_id] = program_storage[p_id].data();
					mem_size += binary_sizes[p_id];
					p_id++;
				}

				if((ret = clGetProgramInfo(ctx->Program[miner_algo], CL_PROGRAM_BINARIES, num_devices * sizeof(char*), all_programs.data(), NULL)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1, "Error %s when calling clGetProgramInfo.", err_to_str(ret));
					return ERR_OCL_API;
				}

				std::ofstream file_stream;
				file_stream.open(cache_file, std::ofstream::out | std::ofstream::binary);
				file_stream.write(all_programs[dev_id], binary_sizes[dev_id]);
				file_stream.close();
				printer::inst()->print_msg(L1, "OpenCL device %u - Precompiled code stored in file %s", ctx->deviceIdx, cache_file.c_str());
			}
		}
		else
		{
			printer::inst()->print_msg(L1, "OpenCL device %u - Load precompiled code from file %s", ctx->deviceIdx, cache_file.c_str());
			std::ostringstream ss;
			ss << clBinFile.rdbuf();
			std::string s = ss.str();

			size_t bin_size = s.size();
			auto data_ptr = s.data();

			cl_int clStatus;
			ctx->Program[miner_algo] = clCreateProgramWithBinary(
				opencl_ctx, 1, &ctx->DeviceID, &bin_size,
				(const unsigned char**)&data_ptr, &clStatus, &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clCreateProgramWithBinary. Try to delete file %s", err_to_str(ret), cache_file.c_str());
				return ERR_OCL_API;
			}
			ret = clBuildProgram(ctx->Program[miner_algo], 1, &ctx->DeviceID, NULL, NULL, NULL);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clBuildProgram. Try to delete file %s", err_to_str(ret), cache_file.c_str());
				return ERR_OCL_API;
			}
		}

		std::vector<std::string> KernelNames = {"cn2", "Blake", "Groestl", "JH", "Skein"};
		if(miner_algo == cryptonight_gpu)
		{
			KernelNames.insert(KernelNames.begin(), "cn1_cn_gpu");
			KernelNames.insert(KernelNames.begin(), "cn0_cn_gpu");
		}
		else
		{
			KernelNames.insert(KernelNames.begin(), "cn1");
			KernelNames.insert(KernelNames.begin(), "cn0");
		}

		// append algorithm number to kernel name
		for(int k = 0; k < 3; k++)
			KernelNames[k] += std::to_string(miner_algo);

		if(miner_algo == cryptonight_gpu)
		{
			KernelNames.push_back(std::string("cn00_cn_gpu") + std::to_string(miner_algo));
		}

		for(int i = 0; i < KernelNames.size(); ++i)
		{
			ctx->Kernels[miner_algo][i] = clCreateKernel(ctx->Program[miner_algo], KernelNames[i].c_str(), &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clCreateKernel for kernel_0 %s.", err_to_str(ret), KernelNames[i].c_str());
				return ERR_OCL_API;
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
	CL_PLATFORM_EXTENSIONS};

const char* const attributeNames[] = {
	"CL_PLATFORM_NAME",
	"CL_PLATFORM_VENDOR",
	"CL_PLATFORM_VERSION",
	"CL_PLATFORM_PROFILE",
	"CL_PLATFORM_EXTENSIONS"};

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

uint32_t getNumPlatforms()
{
	cl_uint num_platforms = 0;
	cl_platform_id* platforms = NULL;
	cl_int clStatus;

	// Get platform and device information
	clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	if(clStatus != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetPlatformIDs for number of platforms.", err_to_str(clStatus));
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
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetPlatformIDs for platform information.", err_to_str(clStatus));
		return ctxVec;
	}

	if((clStatus = clGetDeviceIDs(platforms[index], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceIDs for of devices.", err_to_str(clStatus));
		return ctxVec;
	}

	device_list.resize(num_devices);
	if((clStatus = clGetDeviceIDs(platforms[index], CL_DEVICE_TYPE_GPU, num_devices, device_list.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceIDs for device information.", err_to_str(clStatus));
		return ctxVec;
	}

	for(size_t k = 0; k < num_devices; k++)
	{
		std::vector<char> devVendorVec(1024);
		if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_VENDOR, devVendorVec.size(), devVendorVec.data(), NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get the device vendor name for device %u.", err_to_str(clStatus), k);
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

			ctx.isNVIDIA = isNVIDIADevice;
			ctx.isAMD = isAMDDevice;

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &(ctx.computeUnits), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_MAX_COMPUTE_UNITS for device %u.", err_to_str(clStatus), k);
				continue;
			}

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &(ctx.maxMemPerAlloc), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_MAX_MEM_ALLOC_SIZE for device %u.", err_to_str(clStatus), k);
				continue;
			}

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &(ctx.freeMem), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_GLOBAL_MEM_SIZE for device %u.", err_to_str(clStatus), k);
				continue;
			}

			// the allocation for NVIDIA OpenCL is not limited to 1/4 of the GPU memory per allocation
			if(isNVIDIADevice)
				ctx.maxMemPerAlloc = ctx.freeMem;

			if((clStatus = clGetDeviceInfo(device_list[k], CL_DEVICE_NAME, devNameVec.size(), devNameVec.data(), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_NAME for device %u.", err_to_str(clStatus), k);
				continue;
			}

			std::vector<char> openCLDriverVer(1024);
			if((clStatus = clGetDeviceInfo(device_list[k], CL_DRIVER_VERSION, openCLDriverVer.size(), openCLDriverVer.data(), NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DRIVER_VERSION for device %u.", err_to_str(clStatus), k);
				continue;
			}

			bool isHSAOpenCL = std::string(openCLDriverVer.data()).find("HSA") != std::string::npos;

			// if environment variable GPU_SINGLE_ALLOC_PERCENT is not set we can not allocate the full memory
			ctx.deviceIdx = k;
			ctx.name = std::string(devNameVec.data());
			ctx.DeviceID = device_list[k];
			ctx.interleave = 40;
			printer::inst()->print_msg(L0, "Found OpenCL GPU %s.", ctx.name.c_str());
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
		printer::inst()->print_msg(L0, "WARNING: No OpenCL platform found.");
		return -1;
	}
	cl_platform_id* platforms = NULL;
	cl_int clStatus;

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);

	int platformIndex = -1;
	// Mesa OpenCL is the fallback if no AMD or Apple OpenCL is found
	int mesaPlatform = -1;

	if(clStatus == CL_SUCCESS)
	{
		for(int i = 0; i < numPlatforms; i++)
		{
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
				printer::inst()->print_msg(L0, "Found %s platform index id = %i, name = %s", selectedOpenCLVendor.c_str(), i, platformName.c_str());
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
			printer::inst()->print_msg(L0, "No AMD platform found select Mesa as OpenCL platform");
			platformIndex = mesaPlatform;
		}
	}
	else
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetPlatformIDs for platform information.", err_to_str(clStatus));

	free(platforms);
	return platformIndex;
}

// RequestedDeviceIdxs is a list of OpenCL device indexes
// NumDevicesRequested is number of devices in RequestedDeviceIdxs list
// Returns 0 on success, -1 on stupid params, -2 on OpenCL API error
size_t InitOpenCL(GpuContext* ctx, size_t num_gpus, size_t platform_idx)
{
	cl_int ret;
	cl_uint entries;

	if((ret = clGetPlatformIDs(0, NULL, &entries)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clGetPlatformIDs for number of platforms.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// The number of platforms naturally is the index of the last platform plus one.
	if(entries <= platform_idx)
	{
		printer::inst()->print_msg(L1, "Selected OpenCL platform index %d doesn't exist.", platform_idx);
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
		printer::inst()->print_msg(L1, "Error %s when calling clGetPlatformIDs for platform ID information.", err_to_str(ret));
		return ERR_OCL_API;
	}

	size_t infoSize;
	clGetPlatformInfo(PlatformIDList[platform_idx], CL_PLATFORM_VENDOR, 0, NULL, &infoSize);
	std::vector<char> platformNameVec(infoSize);
	clGetPlatformInfo(PlatformIDList[platform_idx], CL_PLATFORM_VENDOR, infoSize, platformNameVec.data(), NULL);
	std::string platformName(platformNameVec.data());
	if(xmrstak::params::inst().openCLVendor == "AMD" && platformName.find("Advanced Micro Devices") == std::string::npos)
	{
		printer::inst()->print_msg(L1, "WARNING: using non AMD device: %s", platformName.c_str());
	}

	if((ret = clGetDeviceIDs(PlatformIDList[platform_idx], CL_DEVICE_TYPE_GPU, 0, NULL, &entries)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clGetDeviceIDs for number of devices.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Same as the platform index sanity check, except we must check all requested device indexes
	for(int i = 0; i < num_gpus; ++i)
	{
		if(ctx[i].deviceIdx >= entries)
		{
			printer::inst()->print_msg(L1, "Selected OpenCL device index %lu doesn't exist.\n", ctx[i].deviceIdx);
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
		printer::inst()->print_msg(L1, "Error %s when calling clGetDeviceIDs for device ID information.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Indexes sanity checked above
	std::vector<cl_device_id> TempDeviceList(num_gpus, nullptr);

	printer::inst()->print_msg(LDEBUG, "Number of OpenCL GPUs %d", entries);
	for(int i = 0; i < num_gpus; ++i)
	{
		ctx[i].DeviceID = DeviceIDList[ctx[i].deviceIdx];
		TempDeviceList[i] = DeviceIDList[ctx[i].deviceIdx];
	}

	cl_context opencl_ctx = clCreateContext(NULL, num_gpus, TempDeviceList.data(), NULL, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateContext.", err_to_str(ret));
		return ERR_OCL_API;
	}

	const char* fastIntMathV2CL =
#include "./opencl/fast_int_math_v2.cl"
		;
	const char* fastDivHeavyCL =
#include "./opencl/fast_div_heavy.cl"
		;
	const char* cryptonightCL =
#include "./opencl/cryptonight.cl"
		;
	const char* blake256CL =
#include "./opencl/blake256.cl"
		;
	const char* groestl256CL =
#include "./opencl/groestl256.cl"
		;
	const char* jhCL =
#include "./opencl/jh.cl"
		;
	const char* wolfAesCL =
#include "./opencl/wolf-aes.cl"
		;
	const char* wolfSkeinCL =
#include "./opencl/wolf-skein.cl"
		;
	const char* cryptonight_gpu =
#include "./opencl/cryptonight_gpu.cl"
		;

	std::string source_code(cryptonightCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_FAST_INT_MATH_V2"), fastIntMathV2CL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_FAST_DIV_HEAVY"), fastDivHeavyCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_WOLF_AES"), wolfAesCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_WOLF_SKEIN"), wolfSkeinCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_JH"), jhCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_BLAKE256"), blake256CL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_GROESTL256"), groestl256CL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_CN_GPU"), cryptonight_gpu);

	// create a directory  for the OpenCL compile cache
	const std::string cache_dir = xmrstak::params::inst().rootAMDCacheDir;
	create_directory(cache_dir);

	std::vector<std::shared_ptr<InterleaveData>> interleaveData(num_gpus, nullptr);

	for(int i = 0; i < num_gpus; ++i)
	{
		printer::inst()->print_msg(LDEBUG, "OpenCL Init device %d", ctx[i].deviceIdx);
		const size_t devIdx = ctx[i].deviceIdx;
		if(interleaveData.size() <= devIdx)
		{
			interleaveData.resize(devIdx + 1u, nullptr);
		}
		if(!interleaveData[devIdx])
		{
			interleaveData[devIdx].reset(new InterleaveData{});
			interleaveData[devIdx]->lastRunTimeStamp = get_timestamp_ms();
		}
		ctx[i].idWorkerOnDevice = interleaveData[devIdx]->numThreadsOnGPU;
		++interleaveData[devIdx]->numThreadsOnGPU;
		ctx[i].interleaveData = interleaveData[devIdx];
		ctx[i].interleaveData->adjustThreshold = static_cast<double>(ctx[i].interleave) / 100.0;
		ctx[i].interleaveData->startAdjustThreshold = ctx[i].interleaveData->adjustThreshold;
		ctx[i].opencl_ctx = opencl_ctx;

		if((ret = InitOpenCLGpu(ctx->opencl_ctx, &ctx[i], source_code.c_str())) != ERR_SUCCESS)
		{
			return ret;
		}
	}

	return ERR_SUCCESS;
}

size_t XMRSetJob(GpuContext* ctx, uint8_t* input, size_t input_len, uint64_t target, const xmrstak_algo& miner_algo, uint64_t height)
{

	auto& Kernels = ctx->Kernels[miner_algo.Id()];

	cl_int ret;

	if(input_len > 124)
		return ERR_STUPID_PARAMS;

	input[input_len] = 0x01;
	memset(input + input_len + 1, 0, 128 - input_len - 1);

	cl_uint numThreads = ctx->rawIntensity;

	if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->InputBuffer, CL_TRUE, 0, 128, input, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clEnqueueWriteBuffer to fill input buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if((ret = clSetKernelArg(Kernels[0], 0, sizeof(cl_mem), &ctx->InputBuffer)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 0, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Scratchpads
	if((ret = clSetKernelArg(Kernels[0], 1, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 0, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(Kernels[0], 2, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 0, argument 2.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(Kernels[0], 3, sizeof(cl_uint), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 0, argument 3.", err_to_str(ret));
		return (ERR_OCL_API);
	}

	if(miner_algo == cryptonight_gpu)
	{
		// we use an additional cn0 kernel to prepare the scratchpad
		// Scratchpads
		if((ret = clSetKernelArg(Kernels[7], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 0, argument 1.", err_to_str(ret));
			return ERR_OCL_API;
		}

		// States
		if((ret = clSetKernelArg(Kernels[7], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 0, argument 2.", err_to_str(ret));
			return ERR_OCL_API;
		}
	}

	// CN1 Kernel

	if((miner_algo == cryptonight_r) || (miner_algo == cryptonight_r_wow))
	{

		uint32_t PRECOMPILATION_DEPTH = 1;
		constexpr uint64_t height_chunk_size = 25;
		uint64_t height_offset = (height / height_chunk_size) * height_chunk_size;

		// Get new kernel
		cl_program program = xmrstak::amd::CryptonightR_get_program(ctx, miner_algo, height_offset, height_chunk_size, PRECOMPILATION_DEPTH);

		if(program != ctx->ProgramCryptonightR || ctx->last_block_height != height)
		{
			cl_int ret;
			std::string kernel_name = "cn1_cryptonight_r_" + std::to_string(height);
			cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &ret);

			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(LDEBUG, "CryptonightR: clCreateKernel returned error %s", err_to_str(ret));
			}
			else
			{
				cl_kernel old_kernel = Kernels[1];
				if(old_kernel)
					clReleaseKernel(old_kernel);
				Kernels[1] = kernel;
			}
			ctx->ProgramCryptonightR = program;
			ctx->last_block_height = height;
			printer::inst()->print_msg(LDEBUG, "Set height %llu", height);

			// Precompile next program in background
			for(int i = 1; i <= PRECOMPILATION_DEPTH; ++i)
				xmrstak::amd::CryptonightR_get_program(ctx, miner_algo, height_offset + i * height_chunk_size, height_chunk_size, PRECOMPILATION_DEPTH, true);

			printer::inst()->print_msg(LDEBUG, "Thread #%zu updated CryptonightR", ctx->deviceIdx);
		}
		else
		{
			printer::inst()->print_msg(LDEBUG, "Thread #%zu found CryptonightR", ctx->deviceIdx);
		}
	}

	// Scratchpads
	if((ret = clSetKernelArg(Kernels[1], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 1, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(Kernels[1], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 1, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(Kernels[1], 2, sizeof(cl_uint), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 1, argument 2.", err_to_str(ret));
		return (ERR_OCL_API);
	}

	if(miner_algo == cryptonight_monero || miner_algo == cryptonight_aeon || miner_algo == cryptonight_ipbc || miner_algo == cryptonight_stellite || miner_algo == cryptonight_masari || miner_algo == cryptonight_bittube2)
	{
		// Input
		if((ret = clSetKernelArg(Kernels[1], 3, sizeof(cl_mem), &ctx->InputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 1, argument 4(input buffer).", err_to_str(ret));
			return ERR_OCL_API;
		}
	}

	// CN3 Kernel
	// Scratchpads
	if((ret = clSetKernelArg(Kernels[2], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(Kernels[2], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if(miner_algo == cryptonight_gpu)
	{
		// Output
		if((ret = clSetKernelArg(Kernels[2], 2, sizeof(cl_mem), &ctx->OutputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), 2, 2);
			return ERR_OCL_API;
		}

		// Target
		if((ret = clSetKernelArg(Kernels[2], 3, sizeof(cl_ulong), &target)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), 2, 3);
			return ERR_OCL_API;
		}

		// Threads
		if((ret = clSetKernelArg(Kernels[2], 4, sizeof(cl_uint), &numThreads)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 4.", err_to_str(ret));
			return (ERR_OCL_API);
		}
	}
	else
	{
		// Branch 0
		if((ret = clSetKernelArg(Kernels[2], 2, sizeof(cl_mem), ctx->ExtraBuffers + 2)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 2.", err_to_str(ret));
			return ERR_OCL_API;
		}

		// Branch 1
		if((ret = clSetKernelArg(Kernels[2], 3, sizeof(cl_mem), ctx->ExtraBuffers + 3)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 3.", err_to_str(ret));
			return ERR_OCL_API;
		}

		// Branch 2
		if((ret = clSetKernelArg(Kernels[2], 4, sizeof(cl_mem), ctx->ExtraBuffers + 4)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 4.", err_to_str(ret));
			return ERR_OCL_API;
		}

		// Branch 3
		if((ret = clSetKernelArg(Kernels[2], 5, sizeof(cl_mem), ctx->ExtraBuffers + 5)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 5.", err_to_str(ret));
			return ERR_OCL_API;
		}

		// Threads
		if((ret = clSetKernelArg(Kernels[2], 6, sizeof(cl_uint), &numThreads)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel 2, argument 6.", err_to_str(ret));
			return (ERR_OCL_API);
		}

		for(int i = 0; i < 4; ++i)
		{
			// States
			if((ret = clSetKernelArg(Kernels[i + 3], 0, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 0);
				return ERR_OCL_API;
			}

			// Nonce buffer
			if((ret = clSetKernelArg(Kernels[i + 3], 1, sizeof(cl_mem), ctx->ExtraBuffers + (i + 2))) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 1);
				return ERR_OCL_API;
			}

			// Output
			if((ret = clSetKernelArg(Kernels[i + 3], 2, sizeof(cl_mem), &ctx->OutputBuffer)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 2);
				return ERR_OCL_API;
			}

			// Target
			if((ret = clSetKernelArg(Kernels[i + 3], 3, sizeof(cl_ulong), &target)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 3);
				return ERR_OCL_API;
			}

			if((clSetKernelArg(Kernels[i + 3], 4, sizeof(cl_uint), &numThreads)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 4);
				return (ERR_OCL_API);
			}
		}
	}

	return ERR_SUCCESS;
}

uint64_t updateTimings(GpuContext* ctx, const uint64_t t)
{
	// averagingBias = 1.0 - only the last delta time is taken into account
	// averagingBias = 0.5 - the last delta time has the same weight as all the previous ones combined
	// averagingBias = 0.1 - the last delta time has 10% weight of all the previous ones combined
	const double averagingBias = 0.1;

	int64_t t2 = get_timestamp_ms();
	uint64_t runtime = (t2 - t);
	{

		std::lock_guard<std::mutex> g(ctx->interleaveData->mutex);
		// 20000 mean that something went wrong an we reset the average
		if(ctx->interleaveData->avgKernelRuntime == 0.0 || ctx->interleaveData->avgKernelRuntime > 20000.0)
			ctx->interleaveData->avgKernelRuntime = runtime;
		else
			ctx->interleaveData->avgKernelRuntime = ctx->interleaveData->avgKernelRuntime * (1.0 - averagingBias) + (runtime)*averagingBias;
	}
	return runtime;
}

uint64_t interleaveAdjustDelay(GpuContext* ctx, const bool enableAutoAdjustment)
{
	uint64_t t0 = get_timestamp_ms();

	if(ctx->interleaveData->numThreadsOnGPU > 1 && ctx->interleaveData->adjustThreshold > 0.0)
	{
		t0 = get_timestamp_ms();
		std::unique_lock<std::mutex> g(ctx->interleaveData->mutex);

		int64_t delay = 0;
		double dt = 0.0;

		if(t0 > ctx->interleaveData->lastRunTimeStamp)
			dt = static_cast<double>(t0 - ctx->interleaveData->lastRunTimeStamp);

		const double avgRuntime = ctx->interleaveData->avgKernelRuntime;
		const double optimalTimeOffset = avgRuntime * ctx->interleaveData->adjustThreshold;

		// threshold where the the auto adjustment is disabled
		constexpr uint32_t maxDelay = 10;
		constexpr double maxAutoAdjust = 0.05;

		if((dt > 0) && (dt < optimalTimeOffset))
		{
			delay = static_cast<int64_t>((optimalTimeOffset - dt));

			if(enableAutoAdjustment)
			{
				if(ctx->lastDelay == delay && delay > maxDelay)
					ctx->interleaveData->adjustThreshold -= 0.001;
				// if the delay doubled than increase the adjustThreshold
				else if(delay > 1 && ctx->lastDelay * 2 < delay)
					ctx->interleaveData->adjustThreshold += 0.001;
			}
			ctx->lastDelay = delay;

			// this is std::clamp which is available in c++17
			ctx->interleaveData->adjustThreshold = std::max(ctx->interleaveData->adjustThreshold, ctx->interleaveData->startAdjustThreshold - maxAutoAdjust);
			ctx->interleaveData->adjustThreshold = std::min(ctx->interleaveData->adjustThreshold, ctx->interleaveData->startAdjustThreshold + maxAutoAdjust);

			// avoid that the auto adjustment is disable interleaving
			ctx->interleaveData->adjustThreshold = std::max(
				ctx->interleaveData->adjustThreshold,
				0.001);
		}
		delay = std::max(int64_t(0), delay);

		ctx->interleaveData->lastRunTimeStamp = t0 + delay;

		g.unlock();
		if(delay > 0)
		{
			// do not notify the user anymore if we reach a good delay
			if(delay > maxDelay)
				printer::inst()->print_msg(L1, "OpenCL Interleave %u|%u: %u/%.2lf ms - %.1lf",
					ctx->deviceIdx,
					ctx->idWorkerOnDevice,
					static_cast<uint32_t>(delay),
					avgRuntime,
					ctx->interleaveData->adjustThreshold * 100.);

			std::this_thread::sleep_for(std::chrono::milliseconds(delay));
		}
	}

	return t0;
}

size_t XMRRunJob(GpuContext* ctx, cl_uint* HashOutput, const xmrstak_algo& miner_algo)
{
	const auto& Kernels = ctx->Kernels[miner_algo.Id()];

	cl_int ret;
	cl_uint zero = 0;
	size_t BranchNonces[4];
	memset(BranchNonces, 0, sizeof(size_t) * 4);

	size_t g_intensity = ctx->rawIntensity;
	size_t w_size = ctx->workSize;
	size_t g_thd = g_intensity;

	if(ctx->compMode)
	{
		// round up to next multiple of w_size
		g_thd = ((g_intensity + w_size - 1u) / w_size) * w_size;
		// number of global threads must be a multiple of the work group size (w_size)
		assert(g_thd % w_size == 0);
	}

	for(int i = 2; i < 6; ++i)
	{
		if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->ExtraBuffers[i], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), &zero, 0, NULL, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clEnqueueWriteBuffer to zero branch buffer counter %d.", err_to_str(ret), i - 2);
			return ERR_OCL_API;
		}
	}

	if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->OutputBuffer, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(cl_uint), &zero, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clEnqueueWriteBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	size_t Nonce[2] = {ctx->Nonce, 1}, gthreads[2] = {g_thd, 8}, lthreads[2] = {8, 8};
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, Kernels[0], 2, Nonce, gthreads, lthreads, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 0);
		return ERR_OCL_API;
	}

	size_t tmpNonce = ctx->Nonce;

	if(miner_algo == cryptonight_gpu)
	{
		size_t thd = 64;
		size_t intens = g_intensity * thd;
		if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, Kernels[7], 1, 0, &intens, &thd, 0, NULL, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 7);
			return ERR_OCL_API;
		}

		size_t w_size_cn_gpu = w_size * 16;
		size_t g_thd_cn_gpu = g_thd * 16;

		if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, Kernels[1], 1, 0, &g_thd_cn_gpu, &w_size_cn_gpu, 0, NULL, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 1);
			return ERR_OCL_API;
		}
	}
	else
	{
		if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, Kernels[1], 1, &tmpNonce, &g_thd, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 1);
			return ERR_OCL_API;
		}
	}

	size_t  NonceT[2] = {0, ctx->Nonce}, gthreadsT[2] = {8, g_thd}, lthreadsT[2] = {8 , w_size};
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, Kernels[2], 2, NonceT, gthreadsT, lthreadsT, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 2);
			return ERR_OCL_API;
	}

	if(miner_algo != cryptonight_gpu)
	{
		for(int i = 0; i < 4; ++i)
		{
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, Kernels[i + 3], 1, &tmpNonce, &g_thd, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), i + 3);
				return ERR_OCL_API;
			}
		}
	}

	// this call is blocking therefore the access to the results without cl_finish is fine
	if((ret = clEnqueueReadBuffer(ctx->CommandQueues, ctx->OutputBuffer, CL_TRUE, 0, sizeof(cl_uint) * 0x100, HashOutput, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clEnqueueReadBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	auto& numHashValues = HashOutput[0xFF];
	// avoid out of memory read, we have only storage for 0xFF results
	if(numHashValues > 0xFF)
		numHashValues = 0xFF;
	ctx->Nonce += g_intensity;

	return ERR_SUCCESS;
}
