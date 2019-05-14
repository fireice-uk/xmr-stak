/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
  *
  */

#include "jconf.hpp"
#include "params.hpp"

#include "xmrstak/misc/console.hpp"
#include "xmrstak/misc/jext.hpp"
#include "xmrstak/misc/utility.hpp"

#include <algorithm>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifdef _WIN32
#define strcasecmp _stricmp
#include <intrin.h>
#else
#include <cpuid.h>
#endif

using namespace rapidjson;

/*
 * This enum needs to match index in oConfigValues, otherwise we will get a runtime error
 */
enum configEnum
{
	aPoolList,
	sCurrency,
	bTlsSecureAlgo,
	iCallTimeout,
	iNetRetry,
	iGiveUpLimit,
	iVerboseLevel,
	bPrintMotd,
	iAutohashTime,
	bDaemonMode,
	sOutputFile,
	iHttpdPort,
	sHttpLogin,
	sHttpPass,
	bPreferIpv4,
	bAesOverride,
	sUseSlowMem
};

struct configVal
{
	configEnum iName;
	const char* sName;
	Type iType;
};

// Same order as in configEnum, as per comment above
// kNullType means any type
configVal oConfigValues[] = {
	{aPoolList, "pool_list", kArrayType},
	{sCurrency, "currency", kStringType},
	{bTlsSecureAlgo, "tls_secure_algo", kTrueType},
	{iCallTimeout, "call_timeout", kNumberType},
	{iNetRetry, "retry_time", kNumberType},
	{iGiveUpLimit, "giveup_limit", kNumberType},
	{iVerboseLevel, "verbose_level", kNumberType},
	{bPrintMotd, "print_motd", kTrueType},
	{iAutohashTime, "h_print_time", kNumberType},
	{bDaemonMode, "daemon_mode", kTrueType},
	{sOutputFile, "output_file", kStringType},
	{iHttpdPort, "httpd_port", kNumberType},
	{sHttpLogin, "http_login", kStringType},
	{sHttpPass, "http_pass", kStringType},
	{bPreferIpv4, "prefer_ipv4", kTrueType},
	{bAesOverride, "aes_override", kNullType},
	{sUseSlowMem, "use_slow_memory", kStringType}};

constexpr size_t iConfigCnt = (sizeof(oConfigValues) / sizeof(oConfigValues[0]));

xmrstak::coin_selection coins[] = {
	// name, userpool, devpool, default_pool_suggestion
	{"aeon7", {POW(cryptonight_aeon)}, {POW(cryptonight_aeon)}, "mine.aeon-pool.com:5555"},
	{"bbscoin", {POW(cryptonight_aeon)}, {POW(cryptonight_aeon)}, nullptr},
	{"bittube", {POW(cryptonight_bittube2)}, {POW(cryptonight_gpu)}, "mining.bit.tube:13333"},
	{"cryptonight", {POW(cryptonight)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_bittube2", {POW(cryptonight_bittube2)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_masari", {POW(cryptonight_masari)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_haven", {POW(cryptonight_haven)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_heavy", {POW(cryptonight_heavy)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_lite", {POW(cryptonight_lite)}, {POW(cryptonight_aeon)}, nullptr},
	{"cryptonight_lite_v7", {POW(cryptonight_aeon)}, {POW(cryptonight_aeon)}, nullptr},
	{"cryptonight_lite_v7_xor", {POW(cryptonight_ipbc)}, {POW(cryptonight_aeon)}, nullptr},
	{"cryptonight_r", {POW(cryptonight_r)}, {POW(cryptonight_r)}, nullptr},
	{"cryptonight_superfast", {POW(cryptonight_superfast)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_turtle", {POW(cryptonight_turtle)}, {POW(cryptonight_turtle)}, nullptr},
	{"cryptonight_v7", {POW(cryptonight_monero)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_v8", {POW(cryptonight_monero_v8)}, {POW(cryptonight_r)}, nullptr},
	{"cryptonight_v8_double", {POW(cryptonight_v8_double)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_v8_half", {POW(cryptonight_v8_half)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_v8_reversewaltz", {POW(cryptonight_v8_reversewaltz)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_v8_zelerius", {POW(cryptonight_v8_zelerius)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_v7_stellite", {POW(cryptonight_stellite)}, {POW(cryptonight_gpu)}, nullptr},
	{"cryptonight_gpu", {POW(cryptonight_gpu)}, {POW(cryptonight_gpu)}, "pool.ryo-currency.com:3333"},
	{"cryptonight_conceal", {POW(cryptonight_conceal)}, {POW(cryptonight_gpu)}, nullptr},
	{"freehaven", {POW(cryptonight_superfast)}, {POW(cryptonight_gpu)}, nullptr},
	{"graft", {POW(cryptonight_v8_reversewaltz), 12, POW(cryptonight_monero_v8)}, {POW(cryptonight_gpu)}, nullptr},
	{"haven", {POW(cryptonight_haven)}, {POW(cryptonight_gpu)}, nullptr},
	{"lethean", {POW(cryptonight_monero)}, {POW(cryptonight_gpu)}, nullptr},
	{"masari", {POW(cryptonight_v8_half)}, {POW(cryptonight_gpu)}, nullptr},
	{"monero", {POW(cryptonight_r)}, {POW(cryptonight_r)}, "pool.usxmrpool.com:3333"},
	{"qrl", {POW(cryptonight_monero)}, {POW(cryptonight_gpu)}, nullptr},
	{"ryo", {POW(cryptonight_gpu)}, {POW(cryptonight_gpu)}, "pool.ryo-currency.com:3333"},
	{"stellite", {POW(cryptonight_v8_half)}, {POW(cryptonight_gpu)}, nullptr},
	{"turtlecoin", {POW(cryptonight_turtle), 6u, POW(cryptonight_aeon)}, {POW(cryptonight_aeon)}, nullptr},
	{"plenteum", {POW(cryptonight_turtle)}, {POW(cryptonight_turtle)}, nullptr},
	{"zelerius", {POW(cryptonight_v8_zelerius), 7, POW(cryptonight_monero_v8)}, {POW(cryptonight_gpu)}, nullptr},
	{"xcash", {POW(cryptonight_v8_double)}, {POW(cryptonight_gpu)}, nullptr}};

constexpr size_t coin_algo_size = (sizeof(coins) / sizeof(coins[0]));

inline bool checkType(Type have, Type want)
{
	if(want == have)
		return true;
	else if(want == kNullType)
		return true;
	else if(want == kTrueType && have == kFalseType)
		return true;
	else if(want == kFalseType && have == kTrueType)
		return true;
	else
		return false;
}

struct jconf::opaque_private
{
	Document jsonDoc;
	Document jsonDocPools;
	const Value* configValues[iConfigCnt]; //Compile time constant

	opaque_private()
	{
	}
};

jconf::jconf()
{
	prv = new opaque_private();
}

uint64_t jconf::GetPoolCount()
{
	if(prv->configValues[aPoolList]->IsArray())
		return prv->configValues[aPoolList]->Size();
	else
		return 0;
}

bool jconf::GetPoolConfig(size_t id, pool_cfg& cfg)
{
	if(id >= GetPoolCount())
		return false;

	typedef const Value* cval;
	cval jaddr, jlogin, jrigid, jpasswd, jnicehash, jtls, jtlsfp, jwt;
	const Value& oThdConf = prv->configValues[aPoolList]->GetArray()[id];

	/* We already checked presence and types */
	jaddr = GetObjectMember(oThdConf, "pool_address");
	jlogin = GetObjectMember(oThdConf, "wallet_address");
	jrigid = GetObjectMember(oThdConf, "rig_id");
	jpasswd = GetObjectMember(oThdConf, "pool_password");
	jnicehash = GetObjectMember(oThdConf, "use_nicehash");
	jtls = GetObjectMember(oThdConf, "use_tls");
	jtlsfp = GetObjectMember(oThdConf, "tls_fingerprint");
	jwt = GetObjectMember(oThdConf, "pool_weight");

	cfg.sPoolAddr = jaddr->GetString();
	cfg.sWalletAddr = jlogin->GetString();
	cfg.sRigId = jrigid->GetString();
	cfg.sPasswd = jpasswd->GetString();
	cfg.nicehash = jnicehash->GetBool();
	cfg.tls = jtls->GetBool();
	cfg.tls_fingerprint = jtlsfp->GetString();
	cfg.raw_weight = jwt->GetUint64();

	size_t dlt = wt_max - wt_min;
	if(dlt != 0)
	{
		/* Normalise weights between 0 and 9.8 */
		cfg.weight = double(cfg.raw_weight - wt_min) * 9.8;
		cfg.weight /= dlt;
	}
	else /* Special case - user selected same weights for everything */
		cfg.weight = 0.0;
	return true;
}

bool jconf::TlsSecureAlgos()
{
	return prv->configValues[bTlsSecureAlgo]->GetBool();
}

bool jconf::PreferIpv4()
{
	return prv->configValues[bPreferIpv4]->GetBool();
}

uint64_t jconf::GetCallTimeout()
{
	return prv->configValues[iCallTimeout]->GetUint64();
}

uint64_t jconf::GetNetRetry()
{
	return prv->configValues[iNetRetry]->GetUint64();
}

uint64_t jconf::GetGiveUpLimit()
{
	return prv->configValues[iGiveUpLimit]->GetUint64();
}

uint64_t jconf::GetVerboseLevel()
{
	return prv->configValues[iVerboseLevel]->GetUint64();
}

bool jconf::PrintMotd()
{
	return prv->configValues[bPrintMotd]->GetBool();
}

uint64_t jconf::GetAutohashTime()
{
	return prv->configValues[iAutohashTime]->GetUint64();
}

uint16_t jconf::GetHttpdPort()
{
	if(xmrstak::params::inst().httpd_port == xmrstak::params::httpd_port_unset)
		return prv->configValues[iHttpdPort]->GetUint();
	else
		return uint16_t(xmrstak::params::inst().httpd_port);
}

const char* jconf::GetHttpUsername()
{
	return prv->configValues[sHttpLogin]->GetString();
}

const char* jconf::GetHttpPassword()
{
	return prv->configValues[sHttpPass]->GetString();
}

bool jconf::DaemonMode()
{
	return prv->configValues[bDaemonMode]->GetBool();
}

const char* jconf::GetOutputFile()
{
	return prv->configValues[sOutputFile]->GetString();
}

void jconf::cpuid(uint32_t eax, int32_t ecx, int32_t val[4])
{
	memset(val, 0, sizeof(int32_t) * 4);

#ifdef _WIN32
	__cpuidex(val, eax, ecx);
#else
	__cpuid_count(eax, ecx, val[0], val[1], val[2], val[3]);
#endif
}

bool jconf::check_cpu_features()
{
	constexpr int AESNI_BIT = 1 << 25;
	constexpr int SSE2_BIT = 1 << 26;
	int32_t cpu_info[4];
	bool bHaveSse2;

	cpuid(1, 0, cpu_info);

	bHaveAes = (cpu_info[2] & AESNI_BIT) != 0;
	bHaveSse2 = (cpu_info[3] & SSE2_BIT) != 0;

	return bHaveSse2;
}

jconf::slow_mem_cfg jconf::GetSlowMemSetting()
{
	const char* opt = prv->configValues[sUseSlowMem]->GetString();

	if(strcasecmp(opt, "always") == 0)
		return always_use;
	else if(strcasecmp(opt, "no_mlck") == 0)
		return no_mlck;
	else if(strcasecmp(opt, "warn") == 0)
		return print_warning;
	else if(strcasecmp(opt, "never") == 0)
		return never_use;
	else
		return unknown_value;
}

std::string jconf::GetMiningCoin()
{
	if(xmrstak::params::inst().currency.length() > 0)
		return xmrstak::params::inst().currency;
	else
		return prv->configValues[sCurrency]->GetString();
}

void jconf::GetAlgoList(std::string& list)
{
	list.reserve(256);
	for(size_t i = 0; i < coin_algo_size; i++)
	{
		list += "\t- ";
		list += coins[i].coin_name;
		list += "\n";
	}
}

bool jconf::IsOnAlgoList(std::string& needle)
{
	std::transform(needle.begin(), needle.end(), needle.begin(), ::tolower);

	for(size_t i = 0; i < coin_algo_size; i++)
	{
		if(needle == coins[i].coin_name)
			return true;
	}
	return false;
}

const char* jconf::GetDefaultPool(const char* needle)
{
	const char* default_example = "pool.example.com:3333";

	for(size_t i = 0; i < coin_algo_size; i++)
	{
		if(strcmp(needle, coins[i].coin_name) == 0)
		{
			if(coins[i].default_pool != nullptr)
				return coins[i].default_pool;
			else
				return default_example;
		}
	}

	return default_example;
}

bool jconf::parse_file(const char* sFilename, bool main_conf)
{
	FILE* pFile;
	char* buffer;
	size_t flen;

	pFile = fopen(sFilename, "rb");
	if(pFile == NULL)
	{
		printer::inst()->print_msg(L0, "Failed to open config file %s.", sFilename);
		return false;
	}

	fseek(pFile, 0, SEEK_END);
	flen = ftell(pFile);
	rewind(pFile);

	if(flen >= 64 * 1024)
	{
		fclose(pFile);
		printer::inst()->print_msg(L0, "Oversized config file - %s.", sFilename);
		return false;
	}

	if(flen <= 16)
	{
		fclose(pFile);
		printer::inst()->print_msg(L0, "File is empty or too short - %s.", sFilename);
		return false;
	}

	buffer = (char*)malloc(flen + 3);
	if(fread(buffer + 1, flen, 1, pFile) != 1)
	{
		free(buffer);
		fclose(pFile);
		printer::inst()->print_msg(L0, "Read error while reading %s.", sFilename);
		return false;
	}
	fclose(pFile);

	//Replace Unicode BOM with spaces - we always use UTF-8
	unsigned char* ubuffer = (unsigned char*)buffer;
	if(ubuffer[1] == 0xEF && ubuffer[2] == 0xBB && ubuffer[3] == 0xBF)
	{
		buffer[1] = ' ';
		buffer[2] = ' ';
		buffer[3] = ' ';
	}

	buffer[0] = '{';
	buffer[flen] = '}';
	buffer[flen + 1] = '\0';

	Document& root = main_conf ? prv->jsonDoc : prv->jsonDocPools;

	root.Parse<kParseCommentsFlag | kParseTrailingCommasFlag>(buffer, flen + 2);
	free(buffer);

	if(root.HasParseError())
	{
		printer::inst()->print_msg(L0, "JSON config parse error in '%s' (offset %llu): %s",
			sFilename, int_port(root.GetErrorOffset()), GetParseError_En(root.GetParseError()));
		return false;
	}

	if(!root.IsObject())
	{ //This should never happen as we created the root ourselves
		printer::inst()->print_msg(L0, "Invalid config file '%s'. No root?", sFilename);
		return false;
	}

	if(main_conf)
	{
		for(size_t i = 2; i < iConfigCnt; i++)
		{
			if(oConfigValues[i].iName != i)
			{
				printer::inst()->print_msg(L0, "Code error. oConfigValues are not in order.");
				return false;
			}

			prv->configValues[i] = GetObjectMember(root, oConfigValues[i].sName);

			if(prv->configValues[i] == nullptr)
			{
				printer::inst()->print_msg(L0, "Invalid config file '%s'. Missing value \"%s\".", sFilename, oConfigValues[i].sName);
				return false;
			}

			if(!checkType(prv->configValues[i]->GetType(), oConfigValues[i].iType))
			{
				printer::inst()->print_msg(L0, "Invalid config file '%s'. Value \"%s\" has unexpected type.", sFilename, oConfigValues[i].sName);
				return false;
			}
		}
	}
	else
	{
		for(size_t i = 0; i < 2; i++)
		{
			if(oConfigValues[i].iName != i)
			{
				printer::inst()->print_msg(L0, "Code error. oConfigValues are not in order.");
				return false;
			}

			prv->configValues[i] = GetObjectMember(root, oConfigValues[i].sName);

			if(prv->configValues[i] == nullptr)
			{
				printer::inst()->print_msg(L0, "Invalid config file '%s'. Missing value \"%s\".", sFilename, oConfigValues[i].sName);
				return false;
			}

			if(!checkType(prv->configValues[i]->GetType(), oConfigValues[i].iType))
			{
				printer::inst()->print_msg(L0, "Invalid config file '%s'. Value \"%s\" has unexpected type.", sFilename, oConfigValues[i].sName);
				return false;
			}
		}
	}

	return true;
}

bool jconf::parse_config(const char* sFilename, const char* sFilenamePools)
{
	if(!check_cpu_features())
	{
		printer::inst()->print_msg(L0, "CPU support of SSE2 is required.");
		return false;
	}

	if(!parse_file(sFilename, true))
		return false;

	if(!parse_file(sFilenamePools, false))
		return false;

	size_t pool_cnt = prv->configValues[aPoolList]->Size();
	if(pool_cnt == 0)
	{
		printer::inst()->print_msg(L0, "Invalid config file. pool_list must not be empty.");
		return false;
	}

	std::vector<size_t> pool_weights;
	pool_weights.reserve(pool_cnt);

	const char* aPoolValues[] = {"pool_address", "wallet_address", "rig_id", "pool_password", "use_nicehash", "use_tls", "tls_fingerprint", "pool_weight"};
	Type poolValTypes[] = {kStringType, kStringType, kStringType, kStringType, kTrueType, kTrueType, kStringType, kNumberType};

	constexpr size_t pvcnt = sizeof(aPoolValues) / sizeof(aPoolValues[0]);
	for(uint32_t i = 0; i < pool_cnt; i++)
	{
		const Value& oThdConf = prv->configValues[aPoolList]->GetArray()[i];

		if(!oThdConf.IsObject())
		{
			printer::inst()->print_msg(L0, "Invalid config file. pool_list must contain objects.");
			return false;
		}

		for(uint32_t j = 0; j < pvcnt; j++)
		{
			const Value* v;
			if((v = GetObjectMember(oThdConf, aPoolValues[j])) == nullptr)
			{
				printer::inst()->print_msg(L0, "Invalid config file. Pool %u does not have the value %s.", i, aPoolValues[j]);
				return false;
			}

			if(!checkType(v->GetType(), poolValTypes[j]))
			{
				printer::inst()->print_msg(L0, "Invalid config file. Value %s for pool %u has unexpected type.", aPoolValues[j], i);
				return false;
			}
		}

		const Value* jwt = GetObjectMember(oThdConf, "pool_weight");
		size_t wt;
		if(!jwt->IsUint64() || (wt = jwt->GetUint64()) == 0)
		{
			printer::inst()->print_msg(L0, "Invalid pool list for pool %u. Pool weight needs to be an integer larger than zero.", i);
			return false;
		}

		pool_weights.emplace_back(wt);
	}

	wt_max = *std::max_element(pool_weights.begin(), pool_weights.end());
	wt_min = *std::min_element(pool_weights.begin(), pool_weights.end());

	if(!prv->configValues[iCallTimeout]->IsUint64() ||
		!prv->configValues[iNetRetry]->IsUint64() ||
		!prv->configValues[iGiveUpLimit]->IsUint64())
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. call_timeout, retry_time and giveup_limit need to be positive integers.");
		return false;
	}

	if(prv->configValues[iCallTimeout]->GetUint64() < 2 || prv->configValues[iNetRetry]->GetUint64() < 2)
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. call_timeout and retry_time need to be larger than 1 second.");
		return false;
	}

	if(!prv->configValues[iVerboseLevel]->IsUint64() || !prv->configValues[iAutohashTime]->IsUint64())
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. verbose_level and h_print_time need to be positive integers.");
		return false;
	}

	if(!prv->configValues[iHttpdPort]->IsUint() || prv->configValues[iHttpdPort]->GetUint() > 0xFFFF)
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. httpd_port has to be in the range 0 to 65535.");
		return false;
	}

	if(prv->configValues[bAesOverride]->IsBool())
		bHaveAes = prv->configValues[bAesOverride]->GetBool();

	if(!bHaveAes)
		printer::inst()->print_msg(L0, "Your CPU doesn't support hardware AES. Don't expect high hashrates.");

	printer::inst()->set_verbose_level(prv->configValues[iVerboseLevel]->GetUint64());

	if(GetSlowMemSetting() == unknown_value)
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. use_slow_memory must be \"always\", \"no_mlck\", \"warn\" or \"never\"");
		return false;
	}

#ifdef _WIN32
	if(GetSlowMemSetting() == no_mlck)
	{
		printer::inst()->print_msg(L0, "On Windows large pages need mlock. Please use another option.");
		return false;
	}
#endif // _WIN32

	std::string ctmp = GetMiningCoin();
	std::transform(ctmp.begin(), ctmp.end(), ctmp.begin(), ::tolower);

	if(ctmp.length() == 0)
	{
		printer::inst()->print_msg(L0, "You need to specify the coin that you want to mine.");
		return false;
	}

	for(size_t i = 0; i < coin_algo_size; i++)
	{
		if(ctmp == coins[i].coin_name)
		{
			currentCoin = coins[i];
			break;
		}
	}

	if(currentCoin.GetDescription(1).GetMiningAlgo() == invalid_algo)
	{
		std::string cl;
		GetAlgoList(cl);
		printer::inst()->print_msg(L0, "Unrecognised coin '%s', your options are:\n%s", ctmp.c_str(), cl.c_str());
		return false;
	}

	return true;
}
