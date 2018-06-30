#pragma once

#include "xmrstak/misc/environment.hpp"
#include "xmrstak/misc/coinDescription.hpp"
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

	bool parse_config(const char* sFilename, const char* sFilenamePools);

	struct pool_cfg {
		const char* sPoolAddr;
		const char* sWalletAddr;
		const char* sRigId;
		const char* sPasswd;
		bool nicehash;
		bool tls;
		const char* tls_fingerprint;
		size_t raw_weight;
		double weight;
	};

	size_t wt_max;
	size_t wt_min;

	uint64_t GetPoolCount();
	bool GetPoolConfig(size_t id, pool_cfg& cfg);

	enum slow_mem_cfg {
		always_use,
		no_mlck,
		print_warning,
		never_use,
		unknown_value
	};

	bool TlsSecureAlgos();

	inline xmrstak::coin_selection GetCurrentCoinSelection() const { return currentCoin; }

	std::string GetMiningCoin();

	static void GetAlgoList(std::string& list);
	static bool IsOnAlgoList(std::string& needle);
	static const char* GetDefaultPool(const char* needle);

	uint64_t GetVerboseLevel();
	bool PrintMotd();
	uint64_t GetAutohashTime();

	const char* GetOutputFile();

	uint64_t GetCallTimeout();
	uint64_t GetNetRetry();
	uint64_t GetGiveUpLimit();

	uint16_t GetHttpdPort();
	const char* GetHttpUsername();
	const char* GetHttpPassword();

	bool DaemonMode();

	bool PreferIpv4();

	inline bool HaveHardwareAes() { return bHaveAes; }

	static void cpuid(uint32_t eax, int32_t ecx, int32_t val[4]);

	slow_mem_cfg GetSlowMemSetting();

private:
	jconf();

	bool parse_file(const char* sFilename, bool main_conf);

	bool check_cpu_features();
	struct opaque_private;
	opaque_private* prv;

	bool bHaveAes;
	xmrstak::coin_selection currentCoin;
};
