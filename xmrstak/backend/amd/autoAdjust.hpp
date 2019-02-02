
#pragma once

#include "amd_gpu/gpu.hpp"
#include "autoAdjust.hpp"
#include "jconf.hpp"

#include "xmrstak/misc/console.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"

#include <vector>
#include <cstdio>
#include <sstream>
#include <string>
#include <iostream>
#include  <algorithm>

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


namespace xmrstak
{
namespace amd
{

class autoAdjust
{
public:

	autoAdjust()
	{

	}

	/** print the adjusted values if needed
	 *
	 * Routine exit the application and print the adjusted values if needed else
	 * nothing is happened.
	 */
	bool printConfig()
	{
		int platformIndex = getAMDPlatformIdx();

		if(platformIndex == -1)
		{
			printer::inst()->print_msg(L0,"WARNING: No AMD OpenCL platform found. Possible driver issues or wrong vendor driver.");
			return false;
		}

		devVec = getAMDDevices(platformIndex);


		int deviceCount = devVec.size();

		if(deviceCount == 0)
		{
			printer::inst()->print_msg(L0,"WARNING: No AMD device found.");
			return false;
		}

		generateThreadConfig(platformIndex);
		return true;
	}

private:

	void generateThreadConfig(const int platformIndex)
	{
		// load the template of the backend config into a char variable
		const char *tpl =
			#include "./config.tpl"
		;

		configEditor configTpl{};
		configTpl.set( std::string(tpl) );

		constexpr size_t byteToMiB = 1024u * 1024u;

		size_t hashMemSize = std::max(
			cn_select_memory(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo()),
			cn_select_memory(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot())
		);

		std::string conf;
		for(auto& ctx : devVec)
		{
			size_t minFreeMem = 128u * byteToMiB;
			/* 1000 is a magic selected limit, the reason is that more than 2GiB memory
			 * sowing down the memory performance because of TLB cache misses
			 */
			size_t maxThreads = 1000u;
			if(
				ctx.name.compare("gfx901") == 0 ||
				ctx.name.compare("gfx904") == 0 ||
				// APU
				ctx.name.compare("gfx902") == 0 ||
				// UNKNOWN
				ctx.name.compare("gfx900") == 0 ||
				ctx.name.compare("gfx903") == 0 ||
				ctx.name.compare("gfx905") == 0
			)
			{
				/* Increase the number of threads for AMD VEGA gpus.
				 * Limit the number of threads based on the issue: https://github.com/fireice-uk/xmr-stak/issues/5#issuecomment-339425089
				 * to avoid out of memory errors
				 */
				maxThreads = 2024u;
			}

			// NVIDIA optimizations
			if(
				ctx.isNVIDIA && (
					ctx.name.find("P100") != std::string::npos ||
				    ctx.name.find("V100") != std::string::npos
				)
			)
			{
				// do not limit the number of threads
				maxThreads = 40000u;
				minFreeMem = 512u * byteToMiB;
			}

			// check if cryptonight_monero_v8 is selected for the user or dev pool
			bool useCryptonight_v8 =
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_monero_v8 ||
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot() == cryptonight_monero_v8 ||
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(0).GetMiningAlgo() == cryptonight_monero_v8 ||
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(0).GetMiningAlgoRoot() == cryptonight_monero_v8;

			// true for all cryptonight_heavy derivates since we check the user and dev pool
			bool useCryptonight_heavy =
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_heavy ||
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot() == cryptonight_heavy ||
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(0).GetMiningAlgo() == cryptonight_heavy ||
				::jconf::inst()->GetCurrentCoinSelection().GetDescription(0).GetMiningAlgoRoot() == cryptonight_heavy;

			// set strided index to default
			ctx.stridedIndex = 1;

			// nvidia performance is very bad if the scratchpad is not contiguous
			if(ctx.isNVIDIA)
				ctx.stridedIndex = 0;

			// use chunked (4x16byte) scratchpad for all backends. Default `mem_chunk` is `2`
			if(useCryptonight_v8)
				ctx.stridedIndex = 2;
			else if(useCryptonight_heavy)
				ctx.stridedIndex = 3;

			// increase all intensity limits by two for aeon
			if(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_lite)
				maxThreads *= 2u;

			// keep 128MiB memory free (value is randomly chosen) from the max available memory
			const size_t maxAvailableFreeMem = ctx.freeMem - minFreeMem;

			size_t memPerThread = std::min(ctx.maxMemPerAlloc, maxAvailableFreeMem);

			uint32_t numThreads = 1u;
			if(ctx.isAMD)
			{
				numThreads = 2;
				size_t memDoubleThread = maxAvailableFreeMem / numThreads;
				memPerThread = std::min(memPerThread, memDoubleThread);
			}

			// 224byte extra memory is used per thread for meta data
			size_t perThread = hashMemSize + 224u;
			size_t maxIntensity = memPerThread / perThread;
			size_t possibleIntensity = std::min( maxThreads , maxIntensity );
			// map intensity to a multiple of the compute unit count, 8 is the number of threads per work group
			size_t intensity = (possibleIntensity / (8 * ctx.computeUnits)) * ctx.computeUnits * 8;
			// in the case we use two threads per gpu we can be relax and need no multiple of the number of compute units
			if(numThreads == 2)
				intensity = (possibleIntensity / 8) * 8;

			//If the intensity is 0, then it's because the multiple of the unit count is greater than intensity
			if (intensity == 0)
			{
				printer::inst()->print_msg(L0, "WARNING: Auto detected intensity unexpectedly low. Try to set the environment variable GPU_SINGLE_ALLOC_PERCENT.");
				intensity = possibleIntensity;

			}
			if (intensity != 0)
			{
				for(uint32_t thd = 0; thd < numThreads; ++thd)
				{
					conf += "  // gpu: " + ctx.name + std::string("  compute units: ") + std::to_string(ctx.computeUnits) + "\n";
					conf += "  // memory:" + std::to_string(memPerThread / byteToMiB) + "|" +
						std::to_string(ctx.maxMemPerAlloc / byteToMiB) + "|" +  std::to_string(maxAvailableFreeMem / byteToMiB) + " MiB (used per thread|max per alloc|total free)\n";
					// set 8 threads per block (this is a good value for the most gpus)
					conf += std::string("  { \"index\" : ") + std::to_string(ctx.deviceIdx) + ",\n" +
						"    \"intensity\" : " + std::to_string(intensity) + ", \"worksize\" : " + std::to_string(8) + ",\n" +
						"    \"affine_to_cpu\" : false, \"strided_index\" : " + std::to_string(ctx.stridedIndex) + ", \"mem_chunk\" : 2,\n"
						"    \"unroll\" : 8, \"comp_mode\" : true, \"interleave\" : " + std::to_string(ctx.interleave) + "\n" +
						"  },\n";
				}
			}
			else
			{
				printer::inst()->print_msg(L0, "WARNING: Ignore gpu %s, %s MiB free memory is not enough to suggest settings.", ctx.name.c_str(), std::to_string(memPerThread / byteToMiB).c_str());
			}
		}

		configTpl.replace("PLATFORMINDEX",std::to_string(platformIndex));
		configTpl.replace("GPUCONFIG",conf);
		configTpl.write(params::inst().configFileAMD);

		const std::string backendName = xmrstak::params::inst().openCLVendor;
		printer::inst()->print_msg(L0, "%s: GPU (OpenCL) configuration stored in file '%s'", backendName.c_str(), params::inst().configFileAMD.c_str());
	}

	std::vector<GpuContext> devVec;
};

} // namespace amd
} // namespace xmrstak
