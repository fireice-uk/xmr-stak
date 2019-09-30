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
#include "xmrstak/net/msgstruct.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/picosha2/picosha2.hpp"
#include "xmrstak/version.hpp"
#include "xmrstak/backend/cpu/crypto/cryptonight_1.h"
#include "xmrstak/backend/amd/amd_gpu/opencl/RandomX/randomx_run_gfx803.h"
#include "xmrstak/backend/amd/amd_gpu/opencl/RandomX/randomx_run_gfx900.h"
#include "xmrstak/backend/cpu/crypto/randomx/randomx.h"
#include "xmrstak/params.hpp"

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
#endif// _WIN32

#include "gpu.hpp"

cl_mem GpuContext::rx_dataset[32] = {};

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

	/* Some kernel spawn 8 times more threads than the user is configuring.
	 * To give the user the correct maximum work size we divide the hardware specific max by 8.
	 */
	MaximumWorkSize /= 8;

	printer::inst()->print_msg(L1, "Device %lu work size %lu / %lu.", ctx->deviceIdx, ctx->workSize, MaximumWorkSize);

	if(ctx->workSize > MaximumWorkSize)
	{
		ctx->workSize = MaximumWorkSize;
		printer::inst()->print_msg(L1, "Device %lu work size to large, reduce to %lu / %lu.", ctx->deviceIdx, ctx->workSize, MaximumWorkSize);
	}

	const std::string backendName = xmrstak::params::inst().openCLVendor;

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

	// in randomX we have only one algorithm
	auto user_algo = neededAlgorithms[0];


	const size_t dataset_size = getRandomXDatasetSize();

	if(!ctx->rx_dataset[ctx->deviceIdx])
	{
		if(!ctx->datasetHost)
		{
			ctx->rx_dataset[ctx->deviceIdx] = clCreateBuffer(opencl_ctx, CL_MEM_READ_ONLY, dataset_size, nullptr, &ret);
		}
		else {
			void* dataset = getRandomXDataset();
			ctx->rx_dataset[ctx->deviceIdx] = clCreateBuffer(opencl_ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, dataset_size, dataset, &ret);
		}

		if(ret != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX dataset.", err_to_str(ret));
			return ERR_OCL_API;
		}
	}

	ctx->rx_scratchpads = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, (user_algo.Mem() + 64) * g_thd, nullptr, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX scratchpads.", err_to_str(ret));
		return ERR_OCL_API;
	}

	ctx->rx_hashes = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 64 * g_thd, nullptr, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX hashes buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	ctx->rx_entropy = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, (128 + 2560) * g_thd, nullptr, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX entropy buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	if(ctx->gcnAsm)
	{
		ctx->rx_registers = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 256 * g_thd, nullptr, &ret);
		if(ret != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX JIT registers buffer.", err_to_str(ret));
			return ERR_OCL_API;
		}

		ctx->rx_intermediate_programs = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 5120 * g_thd, nullptr, &ret);
		if(ret != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX JIT intermediate programs buffer.", err_to_str(ret));
			return ERR_OCL_API;
		}

		ctx->rx_programs = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 10048 * g_thd, nullptr, &ret);
		if(ret != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX JIT programs buffer.", err_to_str(ret));
			return ERR_OCL_API;
		}
	}
	else {
		ctx->rx_vm_states = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, 2560 * g_thd, nullptr, &ret);
		if(ret != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX VM states buffer.", err_to_str(ret));
			return ERR_OCL_API;
		}
	}

	ctx->rx_rounding = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(uint32_t) * g_thd, nullptr, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create RandomX rounding buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Assume we may find up to 0xFF nonces in one run - it's reasonable
	ctx->OutputBuffer = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 0x100, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "Error %s when calling clCreateBuffer to create output buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	std::vector<char> devVendorVec(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DEVICE_VENDOR, devVendorVec.size(), devVendorVec.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get the device vendor name for device %u.", err_to_str(ret), ctx->deviceIdx);
		return ERR_OCL_API;
	}

	std::string devVendor(devVendorVec.data());
	ctx->vendor = devVendor;

	std::vector<char> devNameVec(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DEVICE_NAME, devNameVec.size(), devNameVec.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DEVICE_NAME for device %u.", err_to_str(ret), ctx->deviceIdx);
		return ERR_OCL_API;
	}
	ctx->name = std::string(devNameVec.data());

	std::vector<char> openCLDriverVer(1024);
	if((ret = clGetDeviceInfo(ctx->DeviceID, CL_DRIVER_VERSION, openCLDriverVer.size(), openCLDriverVer.data(), NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1, "WARNING: %s when calling clGetDeviceInfo to get CL_DRIVER_VERSION for device %u.", err_to_str(ret), ctx->deviceIdx);
		return ERR_OCL_API;
	}

	std::string device_name = ctx->name;
	std::transform(ctx->name.begin(), ctx->name.end(), device_name.begin(), ::toupper);
	ctx->gcn_version = ((device_name == "GFX900") || (device_name == "GFX906")) ? 14 : 12;
	printer::inst()->print_msg(LDEBUG, "AMD: select gcn version %u for '%s'", ctx->gcn_version, device_name.c_str());

	for(const auto miner_algo : neededAlgorithms)
	{
		// scratchpad size for the selected mining algorithm
		size_t hashMemSize = miner_algo.Mem();
		int threadMemMask = miner_algo.Mask();
		int hashIterations = miner_algo.Iter();

		std::string options;
		options += " -DALGO=" + std::to_string(miner_algo.Id());
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

		switch(ctx->workSize)
		{
			case 2:
			case 4:
			case 8:
			case 16:
				break;
			default:
				ctx->workSize = 8;

		}
		options += " -DWORKERS_PER_HASH=" + std::to_string(ctx->workSize);
		options += " -DGCN_VERSION=" + std::to_string(ctx->gcn_version);

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
			ret = clBuildProgram(ctx->Program[miner_algo], 1, &ctx->DeviceID, options.c_str(), NULL, NULL);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "Error %s when calling clBuildProgram. Try to delete file %s", err_to_str(ret), cache_file.c_str());
				return ERR_OCL_API;
			}
		}

		std::vector<std::string> KernelNames = {
			"fillAes1Rx4_scratchpad", "fillAes4Rx4_entropy", "hashAes1Rx4",
			"blake2b_initial_hash", "blake2b_hash_registers_32", "blake2b_hash_registers_64",
			ctx->gcnAsm ? "" : "init_vm", ctx->gcnAsm ? "" : "execute_vm", "find_shares",
			ctx->gcnAsm ? "randomx_jit" : "",
			""
		};
		for(int i = 0; i < KernelNames.size(); ++i)
		{
			if(!KernelNames[i][0])
			{
				printer::inst()->print_msg(LDEBUG, "AMD: skip kernel %i (no name)", i);
				continue;
			}
			printer::inst()->print_msg(LDEBUG, "AMD: prepare kernel %s", KernelNames[i].c_str());
			ctx->rx_kernels[i] = clCreateKernel(ctx->Program[miner_algo], KernelNames[i].c_str(), &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "ERROR AMD: clCreateKernel %s", KernelNames[i].c_str());
				return ERR_OCL_API;
			}
		}
		if(ctx->gcnAsm)
		{
			// Adrenaline drivers on Windows and amdgpu-pro drivers on Linux use ELF header's flags (offset 0x30) to store internal device ID
			// Read it from compiled OpenCL code and substitute this ID into pre-compiled binary to make sure the driver accepts it
			uint32_t elf_header_flags = 0;
			const uint32_t elf_header_flags_offset = 0x30;

			size_t bin_size;
			if(clGetProgramInfo(ctx->Program[miner_algo], CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, NULL) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "ERROR AMD: gcnAsm clGetProgramInfo(CL_PROGRAM_BINARY_SIZES)");
				return ERR_OCL_API;
			}

			std::vector<char> binary_data(bin_size);
			char* tmp[1] = { binary_data.data() };
			if(clGetProgramInfo(ctx->Program[miner_algo], CL_PROGRAM_BINARIES, sizeof(char*), tmp, NULL) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "ERROR AMD: gcnAsm clGetProgramInfo(CL_PROGRAM_BINARIES) bin_size = %u", (uint32_t)bin_size);
				return false;
			}

			if(bin_size >= elf_header_flags_offset + sizeof(uint32_t))
			{
				elf_header_flags = *(uint32_t*)(binary_data.data() + elf_header_flags_offset);
			}

			size_t len = (ctx->gcn_version == 14) ? randomx_run_gfx900_bin_size : randomx_run_gfx803_bin_size;
			unsigned char* binary = (ctx->gcn_version == 14) ? randomx_run_gfx900_bin : randomx_run_gfx803_bin;

			// Set correct internal device ID in the pre-compiled binary
			if(elf_header_flags)
			{
				*(uint32_t*)(binary + elf_header_flags_offset) = elf_header_flags;
			}

			cl_int status;
			ctx->AsmProgram = clCreateProgramWithBinary(ctx->opencl_ctx, 1, &ctx->DeviceID, &len, (const unsigned char**) &binary, &status, &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "ERROR AMD: gcnAsm clCreateProgramWithBinary");
				return ERR_OCL_API;
			}

			ret = clBuildProgram(ctx->AsmProgram, 1, &ctx->DeviceID, options.c_str(), NULL, NULL);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "ERROR AMD: gcnAsm clBuildProgram AsmProgram");
				return ERR_OCL_API;
			}

			ctx->rx_kernels[10] = clCreateKernel(ctx->AsmProgram, "randomx_run", &ret);
			if(ret != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "ERROR AMD: gcnAsm clBuildProgram randomx_run");
				return ERR_OCL_API;
			}
		}

		// fillAes1Rx4_scratchpad
		if((ret = clSetKernelArg(ctx->rx_kernels[0], 0, sizeof(cl_mem), &ctx->rx_hashes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 0, 0);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[0], 1, sizeof(cl_mem), &ctx->rx_scratchpads)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 0, 1);
			return ERR_OCL_API;
		}

		const uint32_t batch_size = g_thd;
		if((ret = clSetKernelArg(ctx->rx_kernels[0], 2, sizeof(uint32_t), &batch_size)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 0, 2);
			return ERR_OCL_API;
		}

		const uint32_t rx_version = (miner_algo == randomX_wow) ? 103 : 104;
		if((ret = clSetKernelArg(ctx->rx_kernels[0], 3, sizeof(uint32_t), &rx_version)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 0, 3);
			return ERR_OCL_API;
		}

		// fillAes4Rx4_entropy
		if((ret = clSetKernelArg(ctx->rx_kernels[1], 0, sizeof(cl_mem), &ctx->rx_hashes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 1, 0);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[1], 1, sizeof(cl_mem), &ctx->rx_entropy)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 1, 1);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[1], 2, sizeof(uint32_t), &batch_size)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 1, 2);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[1], 3, sizeof(uint32_t), &rx_version)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 1, 3);
			return ERR_OCL_API;
		}

		// hashAes1Rx4
		if((ret = clSetKernelArg(ctx->rx_kernels[2], 0, sizeof(cl_mem), &ctx->rx_scratchpads)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 2, 0);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[2], 1, sizeof(cl_mem), ctx->gcnAsm ? &ctx->rx_registers : &ctx->rx_vm_states)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 2, 1);
			return ERR_OCL_API;
		}

		const uint32_t hashOffsetBytes = 192;
		if((ret = clSetKernelArg(ctx->rx_kernels[2], 2, sizeof(uint32_t), &hashOffsetBytes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 2, 2);
			return ERR_OCL_API;
		}

		uint32_t hashStrideBytes;
		if(ctx->gcnAsm)
			hashStrideBytes = 256;
		else
			hashStrideBytes = (miner_algo == randomX_loki) ? RandomX_LokiConfig.ProgramSize * 8 : RandomX_MoneroConfig.ProgramSize * 8;

		if((ret = clSetKernelArg(ctx->rx_kernels[2], 3, sizeof(uint32_t), &hashStrideBytes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 2, 3);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[2], 4, sizeof(uint32_t), &batch_size)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 2, 4);
			return ERR_OCL_API;
		}

		// blake2b_initial_hash
		if((ret = clSetKernelArg(ctx->rx_kernels[3], 0, sizeof(cl_mem), &ctx->rx_hashes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 3, 0);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[3], 1, sizeof(cl_mem), &ctx->InputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 3, 1);
			return ERR_OCL_API;
		}

		// blockTemplateSize is set in RXSetJob()
		// start_nonce is set in RXRunJob()

		// blake2b_hash_registers_32
		if((ret = clSetKernelArg(ctx->rx_kernels[4], 0, sizeof(cl_mem), &ctx->rx_hashes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 4, 0);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[4], 1, sizeof(cl_mem), ctx->gcnAsm ? &ctx->rx_registers : &ctx->rx_vm_states)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 4, 1);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[4], 2, sizeof(uint32_t), &hashStrideBytes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 4, 2);
			return ERR_OCL_API;
		}

		// blake2b_hash_registers_64
		if((ret = clSetKernelArg(ctx->rx_kernels[5], 0, sizeof(cl_mem), &ctx->rx_hashes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 5, 0);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[5], 1, sizeof(cl_mem), ctx->gcnAsm ? &ctx->rx_registers : &ctx->rx_vm_states)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 5, 1);
			return ERR_OCL_API;
		}

		if((ret = clSetKernelArg(ctx->rx_kernels[5], 2, sizeof(uint32_t), &hashStrideBytes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 5, 2);
			return ERR_OCL_API;
		}

		if(!ctx->gcnAsm)
		{
			// init_vm
			if((ret = clSetKernelArg(ctx->rx_kernels[6], 0, sizeof(cl_mem), &ctx->rx_entropy)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 6, 0);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[6], 1, sizeof(cl_mem), &ctx->rx_vm_states)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 6, 1);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[6], 2, sizeof(cl_mem), &ctx->rx_rounding)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 6, 2);
				return ERR_OCL_API;
			}

			// iteration is set in RXRunJob()

			// execute_vm
			if((ret = clSetKernelArg(ctx->rx_kernels[7], 0, sizeof(cl_mem), &ctx->rx_vm_states)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 0);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 1, sizeof(cl_mem), &ctx->rx_rounding)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 1);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 2, sizeof(cl_mem), &ctx->rx_scratchpads)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 2);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 3, sizeof(cl_mem), &ctx->rx_dataset[ctx->deviceIdx])) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 3);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 4, sizeof(uint32_t), &batch_size)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 4);
				return ERR_OCL_API;
			}

			// num_iterations, first, last are set in RXRunJob()
		}

		// find_shares
		if((ret = clSetKernelArg(ctx->rx_kernels[8], 0, sizeof(cl_mem), &ctx->rx_hashes)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 8, 0);
			return ERR_OCL_API;
		}

		// target is set in RXSetJob()
		// start_nonce is set in RXRunJob()

		if((ret = clSetKernelArg(ctx->rx_kernels[8], 3, sizeof(cl_mem), &ctx->OutputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 8, 3);
			return ERR_OCL_API;
		}

		if(ctx->gcnAsm)
		{
			// randomx_jit
			if((ret = clSetKernelArg(ctx->rx_kernels[9], 0, sizeof(cl_mem), &ctx->rx_entropy)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 0);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[9], 1, sizeof(cl_mem), &ctx->rx_registers)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 1);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[9], 2, sizeof(cl_mem), &ctx->rx_intermediate_programs)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 2);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[9], 3, sizeof(cl_mem), &ctx->rx_programs)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 3);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[9], 4, sizeof(uint32_t), &batch_size)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 4);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[9], 5, sizeof(cl_mem), &ctx->rx_rounding)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 5);
				return ERR_OCL_API;
			}

			// iteration is set in RXRunJob()

			// randomx_run
			if((ret = clSetKernelArg(ctx->rx_kernels[10], 0, sizeof(cl_mem), &ctx->rx_dataset[ctx->deviceIdx])) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 0);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[10], 1, sizeof(cl_mem), &ctx->rx_scratchpads)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 1);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[10], 2, sizeof(cl_mem), &ctx->rx_registers)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 2);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[10], 3, sizeof(cl_mem), &ctx->rx_rounding)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 3);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[10], 4, sizeof(cl_mem), &ctx->rx_programs)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 4);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[10], 5, sizeof(uint32_t), &batch_size)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 5);
				return ERR_OCL_API;
			}

			auto PowerOf2 = [](size_t N)
			{
				uint32_t result = 0;
				while (N > 1)
				{
					++result;
					N >>= 1;
				}
				return result;
			};

			const RandomX_ConfigurationBase* rx_conf;
			if(miner_algo == randomX_loki)
				rx_conf = &RandomX_LokiConfig;
			else if(miner_algo == randomX_wow)
				rx_conf = &RandomX_WowneroConfig;
			else if(miner_algo == randomX)
				rx_conf = &RandomX_MoneroConfig;

			const uint32_t rx_parameters =
				(PowerOf2(rx_conf->ScratchpadL1_Size) << 0) |
				(PowerOf2(rx_conf->ScratchpadL2_Size) << 5) |
				(PowerOf2(rx_conf->ScratchpadL3_Size) << 10) |
				(PowerOf2(rx_conf->ProgramIterations) << 15);
			;

			if((ret = clSetKernelArg(ctx->rx_kernels[10], 6, sizeof(uint32_t), &rx_parameters)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1, "clSetKernelArg fail %s %i %i", err_to_str(ret), 10, 6);
				return ERR_OCL_API;
			}
		}
	}

	ctx->Nonce = 0;
	return CL_SUCCESS;
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

			// ifenvironment variable GPU_SINGLE_ALLOC_PERCENT is not set we can not allocate the full memory
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
	// Mesa OpenCL is the fallback ifno AMD or Apple OpenCL is found
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
					// exit ifAMD or Apple platform is found
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

	auto neededAlgorithms = ::jconf::inst()->GetCurrentCoinSelection().GetAllAlgorithms();
	// use the first algo to check for randomX
	auto user_algo = neededAlgorithms[0];

	std::string source_code;

	const char* randomx_constants_wow_h =
		#include "./opencl/RandomX/randomx_constants_wow.h"
	;
	const char* randomx_constants_loki_h =
		#include "./opencl/RandomX/randomx_constants_loki.h"
	;
	const char* randomx_constants_monero_h =
		#include "./opencl/RandomX/randomx_constants_monero.h"
	;
	const char* aesCL =
		#include "./opencl/RandomX/aes.cl"
	;
	const char* fillAes1Rx4CL =
		#include "./opencl/RandomX/fillAes1Rx4.cl"
	;
	const char* blake2bCL =
		#include "./opencl/RandomX/blake2b.cl"
	;
	const char* blake2b_double_blockCL =
		#include "./opencl/RandomX/blake2b_double_block.cl"
	;
	const char* randomx_vmCL =
		#include "./opencl/RandomX/randomx_vm.cl"
	;
	const char* randomx_jitCL =
		#include "./opencl/RandomX/randomx_jit.cl"
	;

	if(user_algo == randomX_wow)
		source_code.append(randomx_constants_wow_h);
	else if(user_algo == randomX_loki)
		source_code.append(randomx_constants_loki_h);
	else if(user_algo == randomX)
		source_code.append(randomx_constants_monero_h);

	source_code.append(std::regex_replace(aesCL, std::regex("#include \"fillAes1Rx4.cl\""), fillAes1Rx4CL));
	source_code.append(std::regex_replace(blake2bCL, std::regex("#include \"blake2b_double_block.cl\""), blake2b_double_blockCL));
	source_code.append(randomx_vmCL);
	source_code.append(randomx_jitCL);

	// create a directory  for the OpenCL compile cache
	const std::string cache_dir = xmrstak::params::inst().rootAMDCacheDir;
	create_directory(cache_dir);

	std::vector<std::shared_ptr<InterleaveData>> interleaveData(num_gpus, nullptr);

	std::map<size_t, bool> overview;

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

		bool known_index = overview.find(devIdx) != overview.end();
		overview[devIdx] = true;
		if(!known_index)
			xmrstak::params::inst().opencl_devices.emplace_back(
				xmrstak::system_entry{ctx[i].vendor + " " + ctx[i].name, ctx[i].rawIntensity});
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
				// ifthe delay doubled than increase the adjustThreshold
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
			// do not notify the user anymore ifwe reach a good delay
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

size_t RXSetJob(GpuContext *ctx, uint8_t *input, size_t input_len, uint64_t target, const uint8_t* seed_hash, const xmrstak_algo& miner_algo)
{
	cl_int ret;
	void* dataset = getRandomXDataset();
	const size_t dataset_size = getRandomXDatasetSize();

	if((memcmp(ctx->rx_dataset_seedhash, seed_hash, sizeof(ctx->rx_dataset_seedhash)) != 0))
	{
		memcpy(ctx->rx_dataset_seedhash, seed_hash, sizeof(ctx->rx_dataset_seedhash));
		//ctx->rx_variant = variant;
		if(!ctx->datasetHost)
		{
			if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->rx_dataset[ctx->deviceIdx], CL_TRUE, 0, dataset_size, dataset, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueWriteBuffer to fill RandomX dataset.", err_to_str(ret));
				return ERR_OCL_API;
			}
		}
	}

	if(input_len < 128)
	{
		memset(input + input_len, 0, 128 - input_len);
	}

	if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->InputBuffer, CL_TRUE, 0, 128, input, 0, nullptr, nullptr)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueWriteBuffer to fill input buffer.", err_to_str(ret));
		return ERR_OCL_API;
	}

	const uint32_t blockTemplateSize = static_cast<uint32_t>(input_len);
	if((ret = clSetKernelArg(ctx->rx_kernels[3], 2, sizeof(uint32_t), &blockTemplateSize)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 3, 2);
		return ERR_OCL_API;
	}

	if((ret = clSetKernelArg(ctx->rx_kernels[8], 1, sizeof(uint64_t), &target)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 8, 1);
		return ERR_OCL_API;
	}

	return ERR_SUCCESS;
}

size_t RXRunJob(GpuContext *ctx, cl_uint *HashOutput, const xmrstak_algo& miner_algo)
{
	const uint32_t g_intensity = static_cast<uint32_t>(ctx->rawIntensity);

	cl_int ret;
	if((ret = clSetKernelArg(ctx->rx_kernels[3], 3, sizeof(uint32_t), &ctx->Nonce)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 3, 3);
		return ERR_OCL_API;
	}

	if((ret = clSetKernelArg(ctx->rx_kernels[8], 2, sizeof(uint32_t), &ctx->Nonce)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 8, 2);
		return ERR_OCL_API;
	}

	size_t globalWorkSize = g_intensity;
	size_t globalWorkSize4 = g_intensity * 4;
	size_t globalWorkSize8 = g_intensity * 8;
	size_t globalWorkSize16 = g_intensity * 16;
	size_t globalWorkSize32 = g_intensity * 32;
	size_t globalWorkSize64 = g_intensity * 64;
	size_t localWorkSize = 64;
	size_t localWorkSize32 = 32;
	size_t localWorkSize16 = 16;

	uint32_t zero = 0;
	if((ret = clEnqueueWriteBuffer(ctx->CommandQueues, ctx->OutputBuffer, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(uint32_t), &zero, 0, nullptr, nullptr)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueWriteBuffer to fetch results.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// blake2b_initial_hash
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[3], 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 3);
		return ERR_OCL_API;
	}

	// fillAes1Rx4_scratchpad
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[0], 1, nullptr, &globalWorkSize4, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 0);
		return ERR_OCL_API;
	}

	uint32_t bfactor = static_cast<uint32_t>(ctx->bfactor);
	if(bfactor > 8)
	{
		bfactor = 8;
	}

	for(uint32_t i = 0; i < RandomX_CurrentConfig.ProgramCount; ++i)
	{
		// fillAes4Rx4_entropy
		if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[1], 1, nullptr, &globalWorkSize4, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 1);
			return ERR_OCL_API;
		}

		if(!ctx->gcnAsm)
		{
			// init_vm
			if((ret = clSetKernelArg(ctx->rx_kernels[6], 3, sizeof(uint32_t), &i)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 6, 3);
				return ERR_OCL_API;
			}

			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[6], 1, nullptr, &globalWorkSize8, &localWorkSize32, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 6);
				return ERR_OCL_API;
			}

			// execute_vm
			uint32_t num_iterations = RandomX_CurrentConfig.ProgramIterations >> bfactor;
			uint32_t first = 1;
			uint32_t last = 0;

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 5, sizeof(uint32_t), &num_iterations)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 5);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 6, sizeof(uint32_t), &first)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 6);
				return ERR_OCL_API;
			}

			if((ret = clSetKernelArg(ctx->rx_kernels[7], 7, sizeof(uint32_t), &last)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 7);
				return ERR_OCL_API;
			}

			for(int j = 0, n = 1 << bfactor; j < n; ++j)
			{
				if(j == n - 1)
				{
					last = 1;
					if((ret = clSetKernelArg(ctx->rx_kernels[7], 7, sizeof(uint32_t), &last)) != CL_SUCCESS)
					{
						printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 7);
						return ERR_OCL_API;
					}
				}

				// execute_vm
				if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[7], 1, nullptr, (ctx->workSize == 16) ? &globalWorkSize16 : &globalWorkSize8, (ctx->workSize == 16) ? &localWorkSize32 : &localWorkSize16, 0, nullptr, nullptr)) != CL_SUCCESS)
				{
					printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 7);
					return ERR_OCL_API;
				}

				if(j == 0)
				{
					first = 0;
					if((ret = clSetKernelArg(ctx->rx_kernels[7], 6, sizeof(uint32_t), &first)) != CL_SUCCESS)
					{
						printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 7, 6);
						return ERR_OCL_API;
					}
				}
			}
		}
		else {
			// randomx_jit
			if((ret = clSetKernelArg(ctx->rx_kernels[9], 6, sizeof(uint32_t), &i)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"clSetKernelArg fail %s %i %i", err_to_str(ret), 9, 6);
				return ERR_OCL_API;
			}

			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[9], 1, nullptr, &globalWorkSize32, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 9);
				return ERR_OCL_API;
			}

			if((ret = clFinish(ctx->CommandQueues)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clFinish.", err_to_str(ret));
				return ERR_OCL_API;
			}

			// randomx_run
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[10], 1, nullptr, &globalWorkSize64, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 10);
				return ERR_OCL_API;
			}
		}

		if(i == RandomX_CurrentConfig.ProgramCount - 1)
		{
			// hashAes1Rx4
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[2], 1, nullptr, &globalWorkSize4, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 2);
				return ERR_OCL_API;
			}

			// blake2b_hash_registers_32
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[4], 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 4);
				return ERR_OCL_API;
			}
		}
		else
		{
			// blake2b_hash_registers_64
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[5], 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 5);
				return ERR_OCL_API;
			}
		}
	}

	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->rx_kernels[8], 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 8);
		return ERR_OCL_API;
	}

	if(clEnqueueReadBuffer(ctx->CommandQueues, ctx->OutputBuffer, CL_FALSE, 0, sizeof(cl_uint) * 0x100, HashOutput, 0, nullptr, nullptr) != CL_SUCCESS)
	{
		return ERR_OCL_API;
	}

	clFinish(ctx->CommandQueues);

	cl_uint& numHashValues = HashOutput[0xFF];

	// avoid out of memory read, we have only storage for 0xFF results
	if(numHashValues > 0xFF)
	{
		numHashValues = 0xFF;
	}

	ctx->Nonce += (uint32_t) g_intensity;

	return ERR_SUCCESS;
}
