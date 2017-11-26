#pragma once

#include "xmrstak/misc/environment.hpp"

#include <string>

namespace xmrstak
{

struct params
{

	static inline params& inst()
	{
		auto& env = environment::inst();
		if(env.pParams == nullptr)
			env.pParams = new params;
		return *env.pParams;
	}

	std::string executablePrefix;
	std::string binaryName;
	bool useAMD;
	bool useNVIDIA;
	bool useCPU;

	std::string poolURL;
	std::string poolPasswd;
	std::string poolUsername;

	std::string currency;

	std::string configFile;
	std::string configFileAMD;
	std::string configFileNVIDIA;
	std::string configFileCPU;

	params() :
		binaryName("xmr-stak"),
		executablePrefix(""),
		useAMD(true),
		useNVIDIA(true),
		useCPU(true),
		configFile("config.txt"),
		configFileAMD("amd.txt"),
		configFileCPU("cpu.txt"),
		configFileNVIDIA("nvidia.txt")
	{}

};

} // namepsace xmrstak
