
#pragma once

#include "autoAdjust.hpp"

#include "jconf.hpp"
#include "nvcc_code/cryptonight.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace xmrstak
{
namespace nvidia
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
		int deviceCount = 0;
		if(cuda_get_devicecount(&deviceCount) == 0)
			return false;
		// evaluate config parameter for if auto adjustment is needed
		for(int i = 0; i < deviceCount; i++)
		{

			nvid_ctx ctx;

			ctx.device_id = i;
			// -1 trigger auto adjustment
			ctx.device_blocks = -1;
			ctx.device_threads = -1;

			// set all device option those marked as auto (-1) to a valid value
#ifndef _WIN32
			ctx.device_bfactor = 0;
			ctx.device_bsleep = 0;
#else
			// windows pass, try to avoid that windows kills the miner if the gpu is blocked for 2 seconds
			ctx.device_bfactor = 6;
			ctx.device_bsleep = 25;
#endif
			if(cuda_get_deviceinfo(&ctx) == 0)
				nvidCtxVec.push_back(ctx);
			else
				printer::inst()->print_msg(L0, "WARNING: NVIDIA setup failed for GPU %d.\n", i);
		}

		generateThreadConfig();
		return true;
	}

  private:
	void generateThreadConfig()
	{
		// load the template of the backend config into a char variable
		const char* tpl =
#include "./config.tpl"
			;

		configEditor configTpl{};
		configTpl.set(std::string(tpl));

		constexpr size_t byte2mib = 1024u * 1024u;
		std::string conf;
		for(auto& ctx : nvidCtxVec)
		{
			if(ctx.device_threads * ctx.device_blocks > 0)
			{
				conf += std::string("  // gpu: ") + ctx.name + " architecture: " + std::to_string(ctx.device_arch[0] * 10 + ctx.device_arch[1]) + "\n";
				conf += std::string("  //      memory: ") + std::to_string(ctx.free_device_memory / byte2mib) + "/" + std::to_string(ctx.total_device_memory / byte2mib) + " MiB\n";
				conf += std::string("  //      smx: ") + std::to_string(ctx.device_mpcount) + "\n";
				conf += std::string("  { \"index\" : ") + std::to_string(ctx.device_id) + ",\n" +
						"    \"threads\" : " + std::to_string(ctx.device_threads) + ", \"blocks\" : " + std::to_string(ctx.device_blocks) + ",\n" +
						"    \"bfactor\" : " + std::to_string(ctx.device_bfactor) + ", \"bsleep\" :  " + std::to_string(ctx.device_bsleep) + ",\n" +
						"    \"affine_to_cpu\" : false, \"sync_mode\" : 3,\n" +
						"    \"mem_mode\" : 1,\n" +
						"  },\n";
			}
		}

		configTpl.replace("GPUCONFIG", conf);
		configTpl.write(params::inst().configFileNVIDIA);
		printer::inst()->print_msg(L0, "NVIDIA: GPU configuration stored in file '%s'", params::inst().configFileNVIDIA.c_str());
	}

	std::vector<nvid_ctx> nvidCtxVec;
};

} // namespace nvidia
} // namespace xmrstak
