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
			env.pParams = new params;
		return *env.pParams;
	}

	std::string executablePrefix;
	std::string binaryName;
	bool useAMD;
	bool AMDCache;
	bool useNVIDIA;
	bool useCPU;
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

	bool allowUAC = true;
	std::string minerArg0;
	std::string minerArgs;

	// block_version >= 0 enable benchmark
	int benchmark_block_version = -1;
	int benchmark_wait_sec = 30;
	int benchmark_work_sec = 60;

	bool no_config_files;
	size_t max_idle_cpu_threads;
	size_t max_cpu_threads_count;
	size_t max_idle_gpu_threads;
	size_t max_gpu_threads_count;
	size_t default_idle_time;

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
		configFileNVIDIA("nvidia.txt"),
		max_idle_cpu_threads((size_t)-1),
		max_cpu_threads_count((size_t)-1),
		max_gpu_threads_count((size_t)-1),
		max_idle_gpu_threads((size_t)-1),
		default_idle_time((size_t)-1),
		no_config_files(false)
	{}

};

} // namespace xmrstak
