#pragma once

#include "xmrstak/misc/console.hpp"
#include "xmrstak/jconf.hpp"

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdint.h>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <map>
#include <array>

#define ERR_SUCCESS (0)
#define ERR_OCL_API (2)
#define ERR_STUPID_PARAMS (1)

struct InterleaveData
{
    std::mutex mutex;

    double adjustThreshold = 0.4;
    double startAdjustThreshold = 0.4;
    double avgKernelRuntime = 0.0;
    uint64_t lastRunTimeStamp = 0;
    uint32_t numThreadsOnGPU = 0;
};

struct GpuContext
{
	/*Input vars*/
	size_t deviceIdx;
	size_t rawIntensity;
	size_t maxRawIntensity;
	size_t workSize;
	int stridedIndex;
	int memChunk;
	int unroll = 0;
	bool isNVIDIA = false;
	bool isAMD = false;
	int compMode;

	/*Output vars*/
	cl_device_id DeviceID;
	cl_command_queue CommandQueues;
	cl_mem InputBuffer;
	cl_mem OutputBuffer;
	cl_mem ExtraBuffers[6];
	std::map<xmrstak_algo, cl_program> Program;
	std::map<xmrstak_algo, std::array<cl_kernel,7>> Kernels;
	size_t freeMem;
	size_t maxMemPerAlloc;
	int computeUnits;
	std::string name;
	std::shared_ptr<InterleaveData> interleaveData;
	uint32_t idWorkerOnDevice = 0u;
	int interleave = 40;
	uint64_t lastDelay = 0;

	uint32_t Nonce;

};

uint32_t getNumPlatforms();
int getAMDPlatformIdx();
std::vector<GpuContext> getAMDDevices(int index);

size_t InitOpenCL(GpuContext* ctx, size_t num_gpus, size_t platform_idx);
size_t XMRSetJob(GpuContext* ctx, uint8_t* input, size_t input_len, uint64_t target, xmrstak_algo miner_algo);
size_t XMRRunJob(GpuContext* ctx, cl_uint* HashOutput, xmrstak_algo miner_algo);
uint64_t interleaveAdjustDelay(GpuContext* ctx, const bool enableAutoAdjustment = true);
uint64_t updateTimings(GpuContext* ctx, const uint64_t t);
