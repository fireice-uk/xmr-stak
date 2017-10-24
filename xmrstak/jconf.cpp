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
#include "xmrstak/misc/console.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
enum configEnum {
	bTlsMode, bTlsSecureAlgo, sTlsFingerprint, sPoolAddr, sWalletAddr, sPoolPwd,sCurrency,
	iCallTimeout, iNetRetry, iGiveUpLimit, iVerboseLevel, iAutohashTime,
	bDaemonMode, sOutputFile, iHttpdPort, bPreferIpv4, bNiceHashMode, bAesOverride, sUseSlowMem };

struct configVal {
	configEnum iName;
	const char* sName;
	Type iType;
};

// Same order as in configEnum, as per comment above
// kNullType means any type
configVal oConfigValues[] = {
	{ bTlsMode, "use_tls", kTrueType },
	{ bTlsSecureAlgo, "tls_secure_algo", kTrueType },
	{ sTlsFingerprint, "tls_fingerprint", kStringType },
	{ sPoolAddr, "pool_address", kStringType },
	{ sWalletAddr, "wallet_address", kStringType },
	{ sPoolPwd, "pool_password", kStringType },
	{ sCurrency, "currency", kStringType },
	{ iCallTimeout, "call_timeout", kNumberType },
	{ iNetRetry, "retry_time", kNumberType },
	{ iGiveUpLimit, "giveup_limit", kNumberType },
	{ iVerboseLevel, "verbose_level", kNumberType },
	{ iAutohashTime, "h_print_time", kNumberType },
	{ bDaemonMode, "daemon_mode", kTrueType },
	{ sOutputFile, "output_file", kStringType },
	{ iHttpdPort, "httpd_port", kNumberType },
	{ bPreferIpv4, "prefer_ipv4", kTrueType },
	{ bNiceHashMode, "nicehash_nonce", kTrueType },
	{ bAesOverride, "aes_override", kNullType },
	{ sUseSlowMem, "use_slow_memory", kStringType }
};

constexpr size_t iConfigCnt = (sizeof(oConfigValues)/sizeof(oConfigValues[0]));

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
	const Value* configValues[iConfigCnt]; //Compile time constant

	opaque_private()
	{
	}
};

jconf::jconf()
{
	prv = new opaque_private();
}

bool jconf::GetTlsSetting()
{
	return prv->configValues[bTlsMode]->GetBool();
}

bool jconf::TlsSecureAlgos()
{
	return prv->configValues[bTlsSecureAlgo]->GetBool();
}

const char* jconf::GetTlsFingerprint()
{
	return prv->configValues[sTlsFingerprint]->GetString();
}

const char* jconf::GetPoolAddress()
{
	auto& poolURL = xmrstak::params::inst().poolURL;
	if(poolURL.empty())
		poolURL = prv->configValues[sPoolAddr]->GetString();
	return poolURL.c_str();
}

const char* jconf::GetPoolPwd()
{
	auto& poolPasswd = xmrstak::params::inst().poolPasswd;
	if(poolPasswd.empty())
		poolPasswd = prv->configValues[sPoolPwd]->GetString();
	return poolPasswd.c_str();

}

const char* jconf::GetWalletAddress()
{
	auto& poolUsername = xmrstak::params::inst().poolUsername;
	if(poolUsername.empty())
		poolUsername = prv->configValues[sWalletAddr]->GetString();
	return poolUsername.c_str();
}

const std::string jconf::GetCurrency()
{
	auto& currency = xmrstak::params::inst().currency;
	if(currency.empty())
		currency = prv->configValues[sCurrency]->GetString();
	if(
#ifndef CONF_NO_XMR
			currency.compare("xmr") != 0
#else
			true
#endif
			&&
#ifndef CONF_NO_AEON
			currency.compare("aeon") != 0
#else
			true
#endif
	)
	{
		printer::inst()->print_msg(L0, "ERROR: Wrong currency selected - '%s'.", currency.c_str());
		win_exit();
	}
	return currency;
}

bool jconf::IsCurrencyXMR()
{
	if(::jconf::inst()->GetCurrency().compare("xmr") == 0)
	{
		return true;
	}
	else
	{
		return false;
	}
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

uint64_t jconf::GetAutohashTime()
{
	return prv->configValues[iAutohashTime]->GetUint64();
}

uint16_t jconf::GetHttpdPort()
{
	return prv->configValues[iHttpdPort]->GetUint();
}

bool jconf::DaemonMode()
{
	return prv->configValues[bDaemonMode]->GetBool();
}

const char* jconf::GetOutputFile()
{
	return prv->configValues[sOutputFile]->GetString();
}

bool jconf::NiceHashMode()
{
	return prv->configValues[bNiceHashMode]->GetBool();
}


void jconf::cpuid(uint32_t eax, int32_t ecx, int32_t val[4])
{
	memset(val, 0, sizeof(int32_t)*4);

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

bool jconf::parse_config(const char* sFilename)
{
	FILE * pFile;
	char * buffer;
	size_t flen;

	if(!check_cpu_features())
	{
		printer::inst()->print_msg(L0, "CPU support of SSE2 is required.");
		return false;
	}

	pFile = fopen(sFilename, "rb");
	if (pFile == NULL)
	{
		printer::inst()->print_msg(L0, "Failed to open config file %s.", sFilename);
		return false;
	}

	fseek(pFile,0,SEEK_END);
	flen = ftell(pFile);
	rewind(pFile);

	if(flen >= 64*1024)
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
	if(fread(buffer+1, flen, 1, pFile) != 1)
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

	prv->jsonDoc.Parse<kParseCommentsFlag|kParseTrailingCommasFlag>(buffer, flen+2);
	free(buffer);

	if(prv->jsonDoc.HasParseError())
	{
		printer::inst()->print_msg(L0, "JSON config parse error(offset %llu): %s",
			int_port(prv->jsonDoc.GetErrorOffset()), GetParseError_En(prv->jsonDoc.GetParseError()));
		return false;
	}


	if(!prv->jsonDoc.IsObject())
	{ //This should never happen as we created the root ourselves
		printer::inst()->print_msg(L0, "Invalid config file. No root?\n");
		return false;
	}

	for(size_t i = 0; i < iConfigCnt; i++)
	{
		if(oConfigValues[i].iName != i)
		{
			printer::inst()->print_msg(L0, "Code error. oConfigValues are not in order.");
			return false;
		}

		prv->configValues[i] = GetObjectMember(prv->jsonDoc, oConfigValues[i].sName);

		if(prv->configValues[i] == nullptr)
		{
			printer::inst()->print_msg(L0, "Invalid config file. Missing value \"%s\".", oConfigValues[i].sName);
			return false;
		}

		if(!checkType(prv->configValues[i]->GetType(), oConfigValues[i].iType))
		{
			printer::inst()->print_msg(L0, "Invalid config file. Value \"%s\" has unexpected type.", oConfigValues[i].sName);
			return false;
		}
	}

	if(!prv->configValues[iCallTimeout]->IsUint64() ||
		!prv->configValues[iNetRetry]->IsUint64() ||
		!prv->configValues[iGiveUpLimit]->IsUint64())
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. call_timeout, retry_time and giveup_limit need to be positive integers.");
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

#ifdef CONF_NO_TLS
	if(prv->configValues[bTlsMode]->GetBool())
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. TLS enabled while the application has been compiled without TLS support.");
		return false;
	}
#endif // CONF_NO_TLS

	/* \todo check in the cpu backend if we have more than 32 worker
	 *  keep in mined that we have change the why how the nonce is calculated (reverse thread index)
	if(NiceHashMode() && GetThreadCount() >= 32)
	{
		printer::inst()->print_msg(L0, "You need to use less than 32 threads in NiceHash mode.");
		return false;
	}
	*/

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

	return true;
}
