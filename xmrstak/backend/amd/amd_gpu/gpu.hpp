#pragma once

#include "xmrstak/misc/console.hpp"

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdint.h>
#include <vector>

#define ERR_SUCCESS (0)
#define ERR_OCL_API (2)
#define ERR_STUPID_PARAMS (1)



struct GpuContext
{
	/*Input vars*/
	size_t deviceIdx;
	size_t rawIntensity;
	size_t workSize;

	/*Output vars*/
	cl_device_id DeviceID;
	cl_command_queue CommandQueues;
	cl_mem InputBuffer;
	cl_mem OutputBuffer;
	cl_mem ExtraBuffers[6];
	cl_program Program;
	cl_kernel Kernels[7];
	size_t freeMem;
	int computeUnits;
	std::string name;

	uint32_t Nonce;

};

uint32_t getNumPlatforms();
int getAMDPlatformIdx();
std::vector<GpuContext> getAMDDevices(int index);

size_t InitOpenCL(GpuContext* ctx, size_t num_gpus, size_t platform_idx);
size_t XMRSetJob(GpuContext* ctx, uint8_t* input, size_t input_len, uint64_t target);
size_t XMRRunJob(GpuContext* ctx, cl_uint* HashOutput);


