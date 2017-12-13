
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
		
		size_t hashMemSize;
		if(::jconf::inst()->IsCurrencyMonero())
		{
			hashMemSize = MONERO_MEMORY;
		}
		else
		{
			hashMemSize = AEON_MEMORY;
		}

		std::string conf;
		int i = 0;
		for(auto& ctx : devVec)
		{
			/*
			  GCN has a hard architectural limit of 64 CUs max (Invalid for other archs though).
			  Reserve 1 CU for the system. 
			*/
			size_t maxComputeUnitsAvailable = std::max(std::min(ctx.computeUnits, 64) - 1, 1);
			//Each CU is 64 threads executed in an SIMT fashion:
			size_t maxThreadsAvailable = maxComputeUnitsAvailable << 6;
			//Keep 128MiB memory free (value is randomly chosen):
			size_t availableMem = ctx.freeMem - (128u * byteToMiB);
			//224byte extra memory is used per thread for meta data
			size_t perThread = hashMemSize + 224u;
			size_t maxIntensity = availableMem / perThread;
			size_t possibleIntensity = std::min(maxThreadsAvailable, maxIntensity);
			//Alias intensity against the smallest work group size possible:
			size_t intensity = (possibleIntensity >> 3) * 8;
			conf += std::string("  // gpu: ") + ctx.name + " memory:" + std::to_string(availableMem / byteToMiB) + "\n";
			conf += std::string("  // compute units: ") + std::to_string(ctx.computeUnits) + "\n";
			//set 8 threads per block (this is a good value for the most gpus)
			//Create two instances if more than one CU is available:
			if (maxComputeUnitsAvailable > 1)
			{
				//Reports seem to indicate splitting a physical device into two works better:
				conf += std::string("  { \"index\" : ") + std::to_string(ctx.deviceIdx) + ",\n" +
				"    \"intensity\" : " + std::to_string((intensity >> 6) * 32) + ", \"worksize\" : " + std::to_string(8) + ",\n" +
				"    \"affine_to_cpu\" : false, \"strided_index\" : false\n" +
				"  },\n";
				intensity -= (intensity >> 6) * 32;
				//Bigger CU count always on the 2nd virtual device, because the system usually takes CU0:
				conf += std::string("  { \"index\" : ") + std::to_string(ctx.deviceIdx) + ",\n" +
				"    \"intensity\" : " + std::to_string(intensity) + ", \"worksize\" : " + std::to_string(8) + ",\n" +
				"    \"affine_to_cpu\" : false, \"strided_index\" : false\n" +
				"  },\n";
			}
			else
			{
				conf += std::string("  { \"index\" : ") + std::to_string(ctx.deviceIdx) + ",\n" +
				"    \"intensity\" : " + std::to_string(intensity) + ", \"worksize\" : " + std::to_string(8) + ",\n" +
				"    \"affine_to_cpu\" : false, \"strided_index\" : true\n" +
				"  },\n";
			}
			++i;
		}

		configTpl.replace("PLATFORMINDEX",std::to_string(platformIndex));
		configTpl.replace("GPUCONFIG",conf);
		configTpl.write(params::inst().configFileAMD);
		printer::inst()->print_msg(L0, "AMD: GPU configuration stored in file '%s'", params::inst().configFileAMD.c_str());
	}

	std::vector<GpuContext> devVec;
};

} // namespace amd
} // namepsace xmrstak
