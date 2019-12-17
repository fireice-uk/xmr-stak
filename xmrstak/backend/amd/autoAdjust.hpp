
#pragma once

#include "amd_gpu/gpu.hpp"
#include "autoAdjust.hpp"
#include "jconf.hpp"

#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/backend/cpu/crypto/cryptonight_1.h"
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
			hashMemSize = std::max(hashMemSize, algo.L3());
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

			// 8 threads per block (this is a good value for the most gpus)
			uint32_t default_workSize = 16;
			size_t minFreeMem = 128u * byteToMiB;

			// disable asm code by default and activate only for RX4XX, RX5XX, Vega, Fiji and VII
			ctx.gcnAsm = false;

			std::string device_name = ctx.name;
			std::transform(ctx.name.begin(), ctx.name.end(), device_name.begin(), ::toupper);

			/* 1000 is a magic selected limit, the reason is that more than 2GiB memory
			 * sowing down the memory performance because of TLB cache misses
			 */
			size_t maxThreads = 1000u;
			uint32_t numThreads = 1u;
			if(
				device_name.compare("GFX901") == 0 ||
				device_name.compare("GFX904") == 0 ||
				// vii
				device_name.compare("GFX906") == 0 ||
				// APU
				device_name.compare("GFX902") == 0 ||
				// UNKNOWN
				device_name.compare("GFX900") == 0 ||
				device_name.compare("GFX903") == 0 ||
				device_name.compare("GFX905") == 0)
			{
				/* Increase the number of threads for AMD VEGA gpus.
				 * Limit the number of threads based on the issue: https://github.com/fireice-uk/xmr-stak/issues/5#issuecomment-339425089
				 * to avoid out of memory errors
				 */
				maxThreads = 2024u;
				ctx.gcnAsm = true;
				numThreads = 2;
			}

			if(
				// RX4XX, RX5XX
				device_name.compare("ELLESMERE") == 0 ||
				device_name.compare("FIJI") == 0
			)
			{
				ctx.gcnAsm = true;
				numThreads = 2;
			}

			// NVIDIA optimizations
			if(
				ctx.isNVIDIA && (device_name.find("P100") != std::string::npos ||
									device_name.find("V100") != std::string::npos))
			{
				// do not limit the number of threads
				maxThreads = 40000u;
			}

			// nvidia performance is very bad if the scratchpad is not contiguous
			if(ctx.isNVIDIA)
				ctx.gcnAsm = false;


			size_t _2MiB = 2llu * 1024 * 1024;
			if(hashMemSize < _2MiB)
			{
				size_t factor = _2MiB / hashMemSize;
				// increase all intensity relative to the original scratchpad size
				maxThreads *= factor;
			}

			// keep 128MiB memory free (value is randomly chosen) from the max available memory
			size_t maxAvailableFreeMem = ctx.freeMem - minFreeMem;

			const size_t dataset_size = getRandomXDatasetSize();
			if(maxAvailableFreeMem <= dataset_size)
				maxAvailableFreeMem = 0;
			else
				maxAvailableFreeMem -= dataset_size;


			size_t memPerThread = std::min(ctx.maxMemPerAlloc, maxAvailableFreeMem);

			// 240byte extra memory is used per thread for meta data
			size_t perThread = hashMemSize + 240u;
			size_t maxIntensity = memPerThread / perThread;
			size_t possibleIntensity = std::min(maxThreads, maxIntensity);
			// map intensity to a multiple of the compute unit count, default_workSize is the number of threads per work group
			size_t intensity = (possibleIntensity / (default_workSize * ctx.computeUnits)) * ctx.computeUnits * default_workSize;
			// in the case we use two threads per gpu we can be relax and need no multiple of the number of compute units
			if(numThreads == 2)
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
							"    \"affine_to_cpu\" : false, \"asm\" : " + (ctx.gcnAsm  ? "true" : "false") + ",\n" +
							"    \"bfactor\" : " + std::to_string(ctx.bfactor) + ", \"interleave\" : " + std::to_string(ctx.interleave) + "\n" +
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
