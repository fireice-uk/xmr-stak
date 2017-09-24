
#pragma once

#include "autoAdjust.hpp"


#include "jconf.h"
#include "../../console.h"
#include "../../ConfigEditor.hpp"
#include "amd_gpu/gpu.h"

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

		devVec = getAMDDevices(0);


		int deviceCount = devVec.size();

        if(deviceCount == 0)
            return false;

 
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

		ConfigEditor configTpl{};
		configTpl.set( std::string(tpl) );

		std::string conf;
		conf += std::string("\"gpu_threads_conf\" :\n[\n");
        int i = 0;
        for(auto& ctx : devVec)
        {
			// use 90% of available memory
			size_t availableMem = (ctx.freeMem * 100u) / 110;
			size_t units = ctx.computeUnits;
			size_t perThread = (size_t(1u)<<21) + 224u;
			size_t max_intensity = availableMem / perThread;
			size_t intensity = std::min( size_t(1000u) , max_intensity );
			conf += std::string(" // gpu: ") + ctx.name + "\n";
            conf += std::string("  { \"index\" : ") + std::to_string(ctx.deviceIdx) + ",\n" +
                "    \"intensity\" : " + std::to_string(intensity) + ", \"worksize\" : " + std::to_string(8) + ",\n" +
                "    \"affine_to_cpu\" : false, \n"
                "  },\n";
            ++i;
        }
		conf += std::string("],\n\n");

		configTpl.replace("PLATFORMINDEX",std::to_string(platformIndex));
		configTpl.replace("NUMGPUS",std::to_string(devVec.size()));
		configTpl.replace("GPUCONFIG",conf);
		configTpl.write("amd.txt");
		printer::inst()->print_msg(L0, "CPU configuration stored in file '%s'", "amd.txt");
    }

    std::vector<GpuContext> devVec;
};

} // namespace amd
} // namepsace xmrstak
