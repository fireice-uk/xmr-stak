#pragma once

#include "xmrstak/misc/environment.hpp"
#include "params.hpp"

#include <stdlib.h>
#include <string>


class jconf
{
public:
	static jconf* inst()
	{
		auto& env = xmrstak::environment::inst();
		if(env.pJconfConfig == nullptr)
			env.pJconfConfig = new jconf;
		return env.pJconfConfig;
	};

	bool parse_config(const char* sFilename = xmrstak::params::inst().configFile.c_str());

	struct thd_cfg {
		bool bDoubleMode;
		bool bNoPrefetch;
		long long iCpuAff;
	};

	enum slow_mem_cfg {
		always_use,
		no_mlck,
		print_warning,
		never_use,
		unknown_value
	};

	bool GetTlsSetting();
	bool TlsSecureAlgos();
	const char* GetTlsFingerprint();

	const char* GetPoolAddress();
	const char* GetPoolPwd();
	const char* GetWalletAddress();
	const std::string GetCurrency();
	bool IsCurrencyMonero();

	uint64_t GetVerboseLevel();
	uint64_t GetAutohashTime();

	const char* GetOutputFile();

	uint64_t GetCallTimeout();
	uint64_t GetNetRetry();
	uint64_t GetGiveUpLimit();

	uint16_t GetHttpdPort();

	bool DaemonMode();

	bool PreferIpv4();

	bool NiceHashMode();

	inline bool HaveHardwareAes() { return bHaveAes; }

	static void cpuid(uint32_t eax, int32_t ecx, int32_t val[4]);

	slow_mem_cfg GetSlowMemSetting();

private:
	jconf();

	bool check_cpu_features();
	struct opaque_private;
	opaque_private* prv;

	bool bHaveAes;
};
