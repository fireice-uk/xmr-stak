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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <regex>
#include <cassert>

#ifdef _WIN32
#include <windows.h>

static inline void port_sleep(size_t sec)
{
	Sleep(sec * 1000);
}
#else
#include <unistd.h>

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
	case CL_INVALID_IMAGE_DESCRIPTOR:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:
		return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:
		return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";
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

	size_t hashMemSize;
	int threadMemMask;
	int hasIterations;
	if(::jconf::inst()->IsCurrencyMonero())
	{
		hashMemSize = MONERO_MEMORY;
		threadMemMask = MONERO_MASK;
		hasIterations = MONERO_ITER;
	}
	else
	{
		hashMemSize = AEON_MEMORY;
		threadMemMask = AEON_MASK;
		hasIterations = AEON_ITER;
	}

	size_t g_thd = ctx->rawIntensity;
	ctx->ExtraBuffers[0] = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, hashMemSize * g_thd, NULL, &ret);
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

	ctx->Program = clCreateProgramWithSource(opencl_ctx, 1, (const char**)&source_code, NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clCreateProgramWithSource on the contents of cryptonight.cl", err_to_str(ret));
		return ERR_OCL_API;
	}

	char options[256];
	snprintf(options, sizeof(options), 
		"-DITERATIONS=%d -DMASK=%d -DWORKSIZE=%llu", hasIterations, threadMemMask, int_port(ctx->workSize));
	ret = clBuildProgram(ctx->Program, 1, &ctx->DeviceID, options, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		size_t len;
		printer::inst()->print_msg(L1,"Error %s when calling clBuildProgram.", err_to_str(ret));

		if((ret = clGetProgramBuildInfo(ctx->Program, ctx->DeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clGetProgramBuildInfo for length of build log output.", err_to_str(ret));
			return ERR_OCL_API;
		}

		char* BuildLog = (char*)malloc(len + 1);
		BuildLog[0] = '\0';

		if((ret = clGetProgramBuildInfo(ctx->Program, ctx->DeviceID, CL_PROGRAM_BUILD_LOG, len, BuildLog, NULL)) != CL_SUCCESS)
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

	cl_build_status status;
	do
	{
		if((ret = clGetProgramBuildInfo(ctx->Program, ctx->DeviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clGetProgramBuildInfo for status of build.", err_to_str(ret));
			return ERR_OCL_API;
		}
		port_sleep(1);
	}
	while(status == CL_BUILD_IN_PROGRESS);

	const char *KernelNames[] = { "cn0", "cn1", "cn2", "Blake", "Groestl", "JH", "Skein" };
	for(int i = 0; i < 7; ++i)
	{
		ctx->Kernels[i] = clCreateKernel(ctx->Program, KernelNames[i], &ret);
		if(ret != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clCreateKernel for kernel %s.", err_to_str(ret), KernelNames[i]);
			return ERR_OCL_API;
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

void PrintDeviceInfo(cl_device_id device)
{
	char queryBuffer[1024];
	int queryInt;
	cl_int clError;
	clError = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(queryBuffer), &queryBuffer, NULL);
	printf("    CL_DEVICE_NAME: %s\n", queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(queryBuffer), &queryBuffer, NULL);
	printf("    CL_DEVICE_VENDOR: %s\n", queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(queryBuffer), &queryBuffer, NULL);
	printf("    CL_DRIVER_VERSION: %s\n", queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(queryBuffer), &queryBuffer, NULL);
	printf("    CL_DEVICE_VERSION: %s\n", queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &queryInt, NULL);
	printf("    CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", queryInt);
}

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
	cl_platform_id * platforms = NULL;
	cl_int clStatus;
	cl_uint num_devices;
	cl_device_id *device_list = NULL;

	uint32_t numPlatforms = getNumPlatforms();

	if(numPlatforms)
	{
		platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numPlatforms);
		clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if(clStatus == CL_SUCCESS)
		{
			clStatus = clGetDeviceIDs( platforms[index], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
			if(clStatus == CL_SUCCESS)
			{
				device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
				clStatus = clGetDeviceIDs( platforms[index], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
				if(clStatus == CL_SUCCESS)
				{
					for (int k = 0; k < num_devices; k++)
					{
						cl_int clError;
						std::vector<char> devVendorVec(1024);
						clError = clGetDeviceInfo(device_list[k], CL_DEVICE_VENDOR, devVendorVec.size(), devVendorVec.data(), NULL);
						if(clStatus == CL_SUCCESS)
						{
							std::string devVendor(devVendorVec.data());
							if( devVendor.find("Advanced Micro Devices") != std::string::npos)
							{
								GpuContext ctx;
								ctx.deviceIdx = k;
								clError = clGetDeviceInfo(device_list[k], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &(ctx.computeUnits), NULL);
								size_t maxMem;
								clError = clGetDeviceInfo(device_list[k], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &(maxMem), NULL);
								clError = clGetDeviceInfo(device_list[k], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &(ctx.freeMem), NULL);
								// if environment variable GPU_SINGLE_ALLOC_PERCENT is not set we can not allocate the full memory
								ctx.freeMem = std::min(ctx.freeMem, maxMem);
								std::vector<char> devNameVec(1024);
								clError = clGetDeviceInfo(device_list[k], CL_DEVICE_NAME, devNameVec.size(), devNameVec.data(), NULL);
								ctx.name = std::string(devNameVec.data());
								printer::inst()->print_msg(L0,"Found OpenCL GPU %s.",ctx.name.c_str());
								ctx.DeviceID = device_list[k];
								ctxVec.push_back(ctx);
							}
						}
						else
							printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceInfo to get the device vendor name.", err_to_str(clStatus));
					}
				}
				else
					printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceIDs for device information.", err_to_str(clStatus));
				free(device_list);
			}
			else
				printer::inst()->print_msg(L1,"WARNING: %s when calling clGetDeviceIDs for of devices.", err_to_str(clStatus));
		}
		else
			printer::inst()->print_msg(L1,"WARNING: %s when calling clGetPlatformIDs for platform information.", err_to_str(clStatus));
	}
	
	free(platforms);

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

	if(clStatus == CL_SUCCESS)
	{
		for (int i = 0; i < numPlatforms; i++) {
			size_t infoSize;
			clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &infoSize);
			std::vector<char> platformNameVec(infoSize);

			clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, infoSize, platformNameVec.data(), NULL);
			std::string platformName(platformNameVec.data());
			if( platformName.find("Advanced Micro Devices") != std::string::npos)
			{
				platformIndex = i;
				printer::inst()->print_msg(L0,"Found AMD platform index id = %i, name = %s",i , platformName.c_str());
				break;
			}
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
	if( platformName.find("Advanced Micro Devices") == std::string::npos)
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
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_WOLF_AES"), wolfAesCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_WOLF_SKEIN"), wolfSkeinCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_JH"), jhCL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_BLAKE256"), blake256CL);
	source_code = std::regex_replace(source_code, std::regex("XMRSTAK_INCLUDE_GROESTL256"), groestl256CL);

	for(int i = 0; i < num_gpus; ++i)
	{
		if((ret = InitOpenCLGpu(opencl_ctx, &ctx[i], source_code.c_str())) != ERR_SUCCESS)
		{
			return ret;
		}
	}

	return ERR_SUCCESS;
}

size_t XMRSetJob(GpuContext* ctx, uint8_t* input, size_t input_len, uint64_t target)
{
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

	if((ret = clSetKernelArg(ctx->Kernels[0], 0, sizeof(cl_mem), &ctx->InputBuffer)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Scratchpads
	if((ret = clSetKernelArg(ctx->Kernels[0], 1, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(ctx->Kernels[0], 2, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 2.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(ctx->Kernels[0], 3, sizeof(cl_ulong), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 0, argument 3.", err_to_str(ret));
		return(ERR_OCL_API);
	}

	// CN2 Kernel

	// Scratchpads
	if((ret = clSetKernelArg(ctx->Kernels[1], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 1, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(ctx->Kernels[1], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 1, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(ctx->Kernels[1], 2, sizeof(cl_ulong), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 1, argument 2.", err_to_str(ret));
		return(ERR_OCL_API);
	}

	// CN3 Kernel
	// Scratchpads
	if((ret = clSetKernelArg(ctx->Kernels[2], 0, sizeof(cl_mem), ctx->ExtraBuffers + 0)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 0.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// States
	if((ret = clSetKernelArg(ctx->Kernels[2], 1, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 1.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 0
	if((ret = clSetKernelArg(ctx->Kernels[2], 2, sizeof(cl_mem), ctx->ExtraBuffers + 2)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 2.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 1
	if((ret = clSetKernelArg(ctx->Kernels[2], 3, sizeof(cl_mem), ctx->ExtraBuffers + 3)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 3.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 2
	if((ret = clSetKernelArg(ctx->Kernels[2], 4, sizeof(cl_mem), ctx->ExtraBuffers + 4)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 4.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Branch 3
	if((ret = clSetKernelArg(ctx->Kernels[2], 5, sizeof(cl_mem), ctx->ExtraBuffers + 5)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 5.", err_to_str(ret));
		return ERR_OCL_API;
	}

	// Threads
	if((ret = clSetKernelArg(ctx->Kernels[2], 6, sizeof(cl_ulong), &numThreads)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel 2, argument 6.", err_to_str(ret));
		return(ERR_OCL_API);
	}

	for(int i = 0; i < 4; ++i)
	{
		// States
		if((ret = clSetKernelArg(ctx->Kernels[i + 3], 0, sizeof(cl_mem), ctx->ExtraBuffers + 1)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 0);
			return ERR_OCL_API;
		}

		// Nonce buffer
		if((ret = clSetKernelArg(ctx->Kernels[i + 3], 1, sizeof(cl_mem), ctx->ExtraBuffers + (i + 2))) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 1);
			return ERR_OCL_API;
		}

		// Output
		if((ret = clSetKernelArg(ctx->Kernels[i + 3], 2, sizeof(cl_mem), &ctx->OutputBuffer)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 2);
			return ERR_OCL_API;
		}

		// Target
		if((ret = clSetKernelArg(ctx->Kernels[i + 3], 3, sizeof(cl_ulong), &target)) != CL_SUCCESS)
		{
			printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 3);
			return ERR_OCL_API;
		}
	}

	return ERR_SUCCESS;
}

size_t XMRRunJob(GpuContext* ctx, cl_uint* HashOutput)
{
	cl_int ret;
	cl_uint zero = 0;
	size_t BranchNonces[4];
	memset(BranchNonces,0,sizeof(size_t)*4);

	size_t g_intensity = ctx->rawIntensity;
	size_t w_size = ctx->workSize;
	// round up to next multiple of w_size
	size_t g_thd = ((g_intensity + w_size - 1u) / w_size) * w_size;
	// number of global threads must be a multiple of the work group size (w_size)
	assert(g_thd%w_size == 0);

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
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[0], 2, Nonce, gthreads, lthreads, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 0);
		return ERR_OCL_API;
	}

	/*for(int i = 1; i < 3; ++i)
	{
		if((ret = clEnqueueNDRangeKernel(*ctx->CommandQueues, ctx->Kernels[i], 1, &ctx->Nonce, &g_thd, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
		{
			Log(LOG_CRITICAL, "Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), i);
			return(ERR_OCL_API);
		}
	}*/

	size_t tmpNonce = ctx->Nonce;
	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[1], 1, &tmpNonce, &g_thd, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printer::inst()->print_msg(L1,"Error %s when calling clEnqueueNDRangeKernel for kernel %d.", err_to_str(ret), 1);
		return ERR_OCL_API;
	}

	if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[2], 2, Nonce, gthreads, lthreads, 0, NULL, NULL)) != CL_SUCCESS)
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
			if((clSetKernelArg(ctx->Kernels[i + 3], 4, sizeof(cl_ulong), BranchNonces + i)) != CL_SUCCESS)
			{
				printer::inst()->print_msg(L1,"Error %s when calling clSetKernelArg for kernel %d, argument %d.", err_to_str(ret), i + 3, 4);
				return(ERR_OCL_API);
			}

			// round up to next multiple of w_size
			BranchNonces[i] = ((BranchNonces[i] + w_size - 1u) / w_size) * w_size;
			// number of global threads must be a multiple of the work group size (w_size)
			assert(BranchNonces[i]%w_size == 0);
			size_t tmpNonce = ctx->Nonce;
			if((ret = clEnqueueNDRangeKernel(ctx->CommandQueues, ctx->Kernels[i + 3], 1, &tmpNonce, BranchNonces + i, &w_size, 0, NULL, NULL)) != CL_SUCCESS)
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
