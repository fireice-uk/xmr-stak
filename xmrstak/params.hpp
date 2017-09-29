#pragma once

#include "xmrstak/misc/environment.hpp"

#include <string>

namespace xmrstak
{

struct Params
{

	static inline Params& inst()
	{
		auto& env = Environment::inst();
		if(env.pParams == nullptr)
			env.pParams = new Params;
		return *env.pParams;
	}

	std::string executablePrefix;
	bool useAMD;
	bool useNVIDIA;
	bool useCPU;

	std::string poolURL;
	std::string poolPasswd;
	std::string poolUsername;

	std::string configFile;
	std::string configFileAMD;
	std::string configFileNVIDIA;
	std::string configFileCPU;

	Params() :
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
