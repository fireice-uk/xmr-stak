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

	std::string executablePrefix = "";
	std::string binaryName = "xmr-stak";
	bool useAMD = true;
	bool useNVIDIA = true;
	bool useCPU = true;

	bool poolUseTls = false;
	std::string poolURL;
	bool userSetPwd = false;
	std::string poolPasswd;
	std::string poolUsername;
	bool nicehashMode = false;

	static constexpr int32_t httpd_port_unset = -1;
	static constexpr int32_t httpd_port_disabled = 0;
	int32_t httpd_port = httpd_port_unset;

	std::string currency;

	std::string configFile = "config.txt";
	std::string configFileAMD = "amd.txt";
	std::string configFileNVIDIA = "nvidia.txt";
	std::string configFileCPU = "cpu.txt";

	bool allowUAC = true;
	std::string minerArg0;
	std::string minerArgs;

	params()
	{}

};

} // namepsace xmrstak
