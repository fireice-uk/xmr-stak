
#pragma once

#include "amd_gpu/gpu.hpp"
#include "autoAdjust.hpp"
#include "jconf.hpp"

#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
			printer::inst()->print_msg(L0, "WARNING: No AMD OpenCL platform found. Possible driver issues or wrong vendor driver.");
			return false;
		}

		devVec = getAMDDevices(platformIndex);

		int deviceCount = devVec.size();

		if(deviceCount == 0)
		{
			printer::inst()->print_msg(L0, "WARNING: No AMD device found.");
			return false;
		}

		generateThreadConfig(platformIndex);
		return true;
	}

  private:
	void generateThreadConfig(const int platformIndex)
	{
		// load the template of the backend config into a char variable
		const char* tpl =
#include "./config.tpl"
			;

		configEditor configTpl{};
		configTpl.set(std::string(tpl));

		constexpr size_t byteToMiB = 1024u * 1024u;

		auto neededAlgorithms = ::jconf::inst()->GetCurrentCoinSelection().GetAllAlgorithms();

		size_t hashMemSize = 0;
		for(const auto algo : neededAlgorithms)
		{
			hashMemSize = std::max(hashMemSize, algo.Mem());
		}

		std::string conf;
		for(auto& ctx : devVec)
		{
			std::string enabledGpus = params::inst().amdGpus;
			bool enabled = true;
			if (!enabledGpus.empty())
			{
				enabled = false;
				std::stringstream ss(enabledGpus);

				int i = -1;
				while (ss >> i)
				{
					if (i == ctx.deviceIdx)
					{
						enabled = true;
						break;
					}

					while (ss.peek() == ',' || ss.peek() == ' ')
						ss.ignore();
				}
			}

			// check if cryptonight_monero_v8 is selected for the user or dev pool
			bool useCryptonight_v8 = (std::find(neededAlgorithms.begin(), neededAlgorithms.end(), cryptonight_monero_v8) != neededAlgorithms.end());

			// true for all cryptonight_heavy derivates since we check the user and dev pool
			bool useCryptonight_heavy = std::find(neededAlgorithms.begin(), neededAlgorithms.end(), cryptonight_heavy) != neededAlgorithms.end();

			// true for cryptonight_gpu as main user pool algorithm
			bool useCryptonight_gpu = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_gpu;

			bool useCryptonight_r = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_r;

			bool useCryptonight_r_wow = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_r_wow;

			// 8 threads per block (this is a good value for the most gpus)
			uint32_t default_workSize = 8;
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
				ctx.name.compare("gfx905") == 0 ||
				// Radeon VII
				ctx.name.compare("gfx906") == 0 ||
				ctx.name.compare("Fiji") == 0)
			{
				/* Increase the number of threads for AMD VEGA gpus.
				 * Limit the number of threads based on the issue: https://github.com/fireice-uk/xmr-stak/issues/5#issuecomment-339425089
				 * to avoid out of memory errors
				 */
				maxThreads = 2024u;

				if(useCryptonight_gpu)
					default_workSize = 16u;
			}

			// NVIDIA optimizations
			if(
				ctx.isNVIDIA && (ctx.name.find("P100") != std::string::npos ||
									ctx.name.find("V100") != std::string::npos))
			{
				// do not limit the number of threads
				maxThreads = 40000u;
				minFreeMem = 512u * byteToMiB;
			}

			// set strided index to default
			ctx.stridedIndex = 1;

			// nvidia performance is very bad if the scratchpad is not contiguous
			if(ctx.isNVIDIA)
				ctx.stridedIndex = 0;

			// use chunked (4x16byte) scratchpad for all backends. Default `mem_chunk` is `2`
			if(useCryptonight_v8 || useCryptonight_r || useCryptonight_r_wow)
				ctx.stridedIndex = 2;
			else if(useCryptonight_heavy)
				ctx.stridedIndex = 3;

			if(hashMemSize < CN_MEMORY)
			{
				size_t factor = CN_MEMORY / hashMemSize;
				// increase all intensity relative to the original scratchpad size
				maxThreads *= factor;
			}

			uint32_t numUnroll = 8;
			uint32_t numThreads = 1u;

			if(useCryptonight_gpu)
			{
				// 6 waves per compute unit are a good value (based on profiling)
				// @todo check again after all optimizations
				maxThreads = ctx.computeUnits * 6 * 8;
				ctx.stridedIndex = 0;
				// do not change unroll for AMD RX5700 but set 2 threads per gpu
				if(ctx.name.compare("gfx1010") == 0)
					numThreads = 2;
				else
					numUnroll = 1;
			}

			// keep 128MiB memory free (value is randomly chosen) from the max available memory
			const size_t maxAvailableFreeMem = ctx.freeMem - minFreeMem;

			size_t memPerThread = std::min(ctx.maxMemPerAlloc, maxAvailableFreeMem);

			if(ctx.isAMD && !useCryptonight_gpu)
			{
				numThreads = 2;
				size_t memDoubleThread = maxAvailableFreeMem / numThreads;
				memPerThread = std::min(memPerThread, memDoubleThread);
			}

			// 240byte extra memory is used per thread for meta data
			size_t perThread = hashMemSize + 240u;
			size_t maxIntensity = memPerThread / perThread;
			size_t possibleIntensity = std::min(maxThreads, maxIntensity);
			// map intensity to a multiple of the compute unit count, default_workSize is the number of threads per work group
			size_t intensity = (possibleIntensity / (default_workSize * ctx.computeUnits)) * ctx.computeUnits * default_workSize;

			size_t computeUnitUtilization = ((possibleIntensity * 100)  / (default_workSize * ctx.computeUnits)) % 100;
			// in the case we use two threads per gpu or if we can utilize over 75% of the compute units
			// we can be relax and need no multiple of the number of compute units
			if(numThreads == 2 || computeUnitUtilization >= 75)
				intensity = (possibleIntensity / default_workSize) * default_workSize;

			//If the intensity is 0, then it's because the multiple of the unit count is greater than intensity
			if(intensity == 0)
			{
				printer::inst()->print_msg(L0, "WARNING: Auto detected intensity unexpectedly low. Try to set the environment variable GPU_SINGLE_ALLOC_PERCENT.");
				intensity = possibleIntensity;
			}
			if(intensity != 0)
			{
				if (!enabled)
					conf += "/* Disabled\n";

				for(uint32_t thd = 0; thd < numThreads; ++thd)
				{
					conf += "  // gpu: " + ctx.name + std::string("  compute units: ") + std::to_string(ctx.computeUnits) + "\n";
					conf += "  // memory:" + std::to_string(memPerThread / byteToMiB) + "|" +
							std::to_string(ctx.maxMemPerAlloc / byteToMiB) + "|" + std::to_string(maxAvailableFreeMem / byteToMiB) + " MiB (used per thread|max per alloc|total free)\n";
					conf += std::string("  { \"index\" : ") + std::to_string(ctx.deviceIdx) + ",\n" +
							"    \"intensity\" : " + std::to_string(intensity) + ", \"worksize\" : " + std::to_string(default_workSize) + ",\n" +
							"    \"affine_to_cpu\" : false, \"strided_index\" : " + std::to_string(ctx.stridedIndex) + ", \"mem_chunk\" : 2,\n"
																													   "    \"unroll\" : " +
							std::to_string(numUnroll) + ", \"comp_mode\" : true, \"interleave\" : " + std::to_string(ctx.interleave) + "\n" +
							"  },\n";
				}

				if (!enabled)
					conf += "*/\n";
			}
			else
			{
				printer::inst()->print_msg(L0, "WARNING: Ignore gpu %s, %s MiB free memory is not enough to suggest settings.", ctx.name.c_str(), std::to_string(memPerThread / byteToMiB).c_str());
			}
		}

		configTpl.replace("PLATFORMINDEX", std::to_string(platformIndex));
		configTpl.replace("GPUCONFIG", conf);
		configTpl.write(params::inst().configFileAMD);

		const std::string backendName = xmrstak::params::inst().openCLVendor;
		printer::inst()->print_msg(L0, "%s: GPU (OpenCL) configuration stored in file '%s'", backendName.c_str(), params::inst().configFileAMD.c_str());
	}

	std::vector<GpuContext> devVec;
};

} // namespace amd
} // namespace xmrstak
