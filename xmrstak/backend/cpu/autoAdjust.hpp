#pragma once

#include "jconf.hpp"

#include "xmrstak/backend/cpu/cpuType.hpp"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif // _WIN32

namespace xmrstak
{
namespace cpu
{

class autoAdjust
{
  public:
	bool printConfig()
	{
		auto neededAlgorithms = ::jconf::inst()->GetCurrentCoinSelection().GetAllAlgorithms();

		size_t hashMemSize = 0;
		for(const auto algo : neededAlgorithms)
		{
			hashMemSize = std::max(hashMemSize, algo.Mem());
		}
		const size_t hashMemSizeKB = hashMemSize / 1024u;

		const size_t halfHashMemSizeKB = hashMemSizeKB / 2u;

		configEditor configTpl{};

		// load the template of the backend config into a char variable
		const char* tpl =
#include "./config.tpl"
			;
		configTpl.set(std::string(tpl));

		std::string conf;

		// if cryptonight_gpu is used we will disable cpu mining but provide a inactive config
		bool useCryptonight_gpu = ::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo() == cryptonight_gpu;

		if(useCryptonight_gpu)
		{
			printer::inst()->print_msg(L0, "WARNING: CPU mining will be disabled because cryptonight_gpu is not suitable for CPU mining. You can uncomment the auto generated config in %s to enable CPU mining.", params::inst().configFileCPU.c_str());
			conf += "/*\n//CPU config is disabled by default because cryptonight_gpu is not suitable for CPU mining.\n";
		}
		if(!detectL3Size() || L3KB_size < halfHashMemSizeKB || L3KB_size > (halfHashMemSizeKB * 2048u))
		{
			if(L3KB_size < halfHashMemSizeKB || L3KB_size > (halfHashMemSizeKB * 2048))
				printer::inst()->print_msg(L0, "Autoconf failed: L3 size sanity check failed - %u KB.", L3KB_size);

			conf += std::string("    { \"low_power_mode\" : false, \"no_prefetch\" : true,  \"asm\" : \"off\", \"affine_to_cpu\" : false },\n");
			printer::inst()->print_msg(L0, "Autoconf FAILED. Create config for a single thread. Please try to add new ones until the hashrate slows down.");
		}
		else
		{
			printer::inst()->print_msg(L0, "Autoconf L3 size detected at %u KB.", L3KB_size);

			detectCPUConf();

			printer::inst()->print_msg(L0, "Autoconf core count detected as %u on %s.", corecnt,
				linux_layout ? "Linux" : "Windows");

			uint32_t aff_id = 0;
			for(uint32_t i = 0; i < corecnt; i++)
			{
				bool double_mode;

				if(L3KB_size <= 0)
					break;

				double_mode = L3KB_size / hashMemSizeKB > (int32_t)(corecnt - i);

				conf += std::string("    { \"low_power_mode\" : ");
				conf += std::string(double_mode ? "true" : "false");
				conf += std::string(", \"no_prefetch\" : true, \"asm\" : \"auto\", \"affine_to_cpu\" : ");
				conf += std::to_string(aff_id);
				conf += std::string(" },\n");

				if(!linux_layout || old_amd)
				{
					aff_id += 2;

					if(aff_id >= corecnt)
						aff_id = 1;
				}
				else
					aff_id++;

				if(double_mode)
					L3KB_size -= hashMemSizeKB * 2u;
				else
					L3KB_size -= hashMemSizeKB;
			}
		}

		if(useCryptonight_gpu)
			conf += "*/\n";

		configTpl.replace("CPUCONFIG", conf);
		configTpl.write(params::inst().configFileCPU);
		printer::inst()->print_msg(L0, "CPU configuration stored in file '%s'", params::inst().configFileCPU.c_str());

		return true;
	}

  private:
	bool detectL3Size()
	{
		int32_t cpu_info[4];
		char cpustr[13] = {0};

		::jconf::cpuid(0, 0, cpu_info);
		memcpy(cpustr, &cpu_info[1], 4);
		memcpy(cpustr + 4, &cpu_info[3], 4);
		memcpy(cpustr + 8, &cpu_info[2], 4);

		if(strcmp(cpustr, "GenuineIntel") == 0)
		{
			::jconf::cpuid(4, 3, cpu_info);

			if(get_masked(cpu_info[0], 7, 5) != 3)
			{
				printer::inst()->print_msg(L0, "Autoconf failed: Couldn't find L3 cache page.");
				return false;
			}

			L3KB_size = ((get_masked(cpu_info[1], 31, 22) + 1) * (get_masked(cpu_info[1], 21, 12) + 1) *
							(get_masked(cpu_info[1], 11, 0) + 1) * (cpu_info[2] + 1)) /
						1024;

			return true;
		}
		else if(strcmp(cpustr, "AuthenticAMD") == 0)
		{
			::jconf::cpuid(0x80000006, 0, cpu_info);

			L3KB_size = get_masked(cpu_info[3], 31, 18) * 512;

			::jconf::cpuid(1, 0, cpu_info);

			if(getModel().family < 0x17) //0x17h is Zen
				old_amd = true;

			return true;
		}
		else
		{
			printer::inst()->print_msg(L0, "Autoconf failed: Unknown CPU type: %s.", cpustr);
			return false;
		}
	}

	void detectCPUConf()
	{
#ifdef _WIN32
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		corecnt = info.dwNumberOfProcessors;
		linux_layout = false;
#else
		corecnt = sysconf(_SC_NPROCESSORS_ONLN);
		linux_layout = true;
#endif // _WIN32
	}

	int32_t L3KB_size = 0;
	uint32_t corecnt;
	bool old_amd = false;
	bool linux_layout;
};

} // namespace cpu
} // namespace xmrstak
