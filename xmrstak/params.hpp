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

	std::string configFile;
	std::string configFileAMD;
	std::string configFileNVIDIA;
	std::string configFileCPU;

	bool allowUAC = true;
	std::string minerArg0;
	std::string minerArgs;

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
