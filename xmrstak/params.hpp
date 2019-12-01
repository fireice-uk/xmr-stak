#pragma once

#include "xmrstak/misc/environment.hpp"
#include "xmrstak/misc/home_dir.hpp"

#include <string>

namespace xmrstak
{

struct params
{

	static inline params& inst()
	{
		auto& env = environment::inst();
		if(env.pParams == nullptr)
		{
			std::unique_lock<std::mutex> lck(env.update);
			if(env.pParams == nullptr)
				env.pParams = new params;
		}
		return *env.pParams;
	}

	std::string executablePrefix;
	std::string binaryName;
	bool useAMD;
	bool AMDCache;
	bool useNVIDIA;
	bool useCPU;
	std::string amdGpus;
	std::string nvidiaGpus;
	// user selected OpenCL vendor
	std::string openCLVendor;

	bool poolUseTls = false;
	std::string poolURL;
	bool userSetPwd = false;
	std::string poolPasswd;
	bool userSetRigid = false;
	std::string poolRigid;
	std::string poolUsername;
	bool nicehashMode = false;

	static constexpr int32_t httpd_port_unset = -1;
	static constexpr int32_t httpd_port_disabled = 0;
	int32_t httpd_port = httpd_port_unset;

	std::string currency;

	std::string configFile;
	std::string configFilePools;
	std::string configFileAMD;
	std::string rootAMDCacheDir;
	std::string configFileNVIDIA;
	std::string configFileCPU;

	std::string outputFile;
	int h_print_time = -1;

	bool allowUAC = true;
	std::string minerArg0;
	std::string minerArgs;

	// block_version >= 0 enable benchmark
	int benchmark_block_version = -1;
	int benchmark_wait_sec = 30;
	int benchmark_work_sec = 60;

	params() :
		binaryName("xmr-stak"),
		executablePrefix(""),
		useAMD(true),
		AMDCache(true),
		useNVIDIA(true),
		useCPU(true),
		openCLVendor("AMD"),
		configFile("config.txt"),
		configFilePools("pools.txt"),
		configFileAMD("amd.txt"),
		rootAMDCacheDir(get_home() + "/.openclcache/"),
		configFileCPU("cpu.txt"),
		configFileNVIDIA("nvidia.txt")
	{
	}
};

} // namespace xmrstak
