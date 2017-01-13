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
  */

#include "jconf.h"
#include "console.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "jext.h"
#include "console.h"

using namespace rapidjson;

/*
 * This enum needs to match index in oConfigValues, otherwise we will get a runtime error
 */
enum configEnum { iCpuThreadNum, aCpuThreadsConf, sUseSlowMem, sPoolAddr,
	sWalletAddr, sPoolPwd, iCallTimeout, iNetRetry, iVerboseLevel, iAutohashTime, bPreferIpv4 };

struct configVal {
	configEnum iName;
	const char* sName;
	Type iType;
};

//Same order as in configEnum, as per comment above
configVal oConfigValues[] = {
	{ iCpuThreadNum, "cpu_thread_num", kNumberType },
	{ aCpuThreadsConf, "cpu_threads_conf", kArrayType },
	{ sUseSlowMem, "use_slow_memory", kStringType },
	{ sPoolAddr, "pool_address", kStringType },
	{ sWalletAddr, "wallet_address", kStringType },
	{ sPoolPwd, "pool_password", kStringType },
	{ iCallTimeout, "call_timeout", kNumberType },
	{ iNetRetry, "retry_time", kNumberType },
	{ iVerboseLevel, "verbose_level", kNumberType },
	{ iAutohashTime, "h_print_time", kNumberType },
	{ bPreferIpv4, "prefer_ipv4", kTrueType }
};

constexpr size_t iConfigCnt = (sizeof(oConfigValues)/sizeof(oConfigValues[0]));

inline bool checkType(Type have, Type want)
{
	if(want == have)
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

jconf* jconf::oInst = nullptr;

jconf::jconf()
{
	prv = new opaque_private();
}

bool jconf::GetThreadConfig(size_t id, thd_cfg &cfg)
{
	if(id >= prv->configValues[aCpuThreadsConf]->Size())
		return false;

	const Value& oThdConf = prv->configValues[aCpuThreadsConf]->GetArray()[id];

	if(!oThdConf.IsObject())
		return false;

	const Value *mode, *aff;
	mode = GetObjectMember(oThdConf, "low_power_mode");
	aff = GetObjectMember(oThdConf, "affine_to_cpu");

	if(mode == nullptr || aff == nullptr)
		return false;

	if(!mode->IsBool())
		return false;

	if(!aff->IsNumber() && !aff->IsBool())
		return false;

	if(aff->IsNumber() && aff->GetInt64() < 0)
		return false;

	cfg.bDoubleMode = mode->GetBool();
	if(aff->IsNumber())
		cfg.iCpuAff = aff->GetInt64();
	else
		cfg.iCpuAff = -1;

	return true;
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

const char* jconf::GetPoolAddress()
{
	return prv->configValues[sPoolAddr]->GetString();
}

const char* jconf::GetPoolPwd()
{
	return prv->configValues[sPoolPwd]->GetString();
}

const char* jconf::GetWalletAddress()
{
	return prv->configValues[sWalletAddr]->GetString();
}

bool jconf::PreferIpv4()
{
	return prv->configValues[bPreferIpv4]->GetBool();
}

size_t jconf::GetThreadCount()
{
	return prv->configValues[aCpuThreadsConf]->Size();
}

uint64_t jconf::GetCallTimeout()
{
	return prv->configValues[iCallTimeout]->GetUint64();
}

uint64_t jconf::GetNetRetry()
{
	return prv->configValues[iNetRetry]->GetUint64();
}

uint64_t jconf::GetVerboseLevel()
{
	return prv->configValues[iVerboseLevel]->GetUint64();
}

uint64_t jconf::GetAutohashTime()
{
	return prv->configValues[iAutohashTime]->GetUint64();
}

bool jconf::parse_config(const char* sFilename)
{
	FILE * pFile;
	char * buffer;
	size_t flen;

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
	if(buffer[1] == 0xEF && buffer[2] == 0xBB && buffer[3] == 0xBF)
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

	size_t n_thd = prv->configValues[aCpuThreadsConf]->Size();
	if(prv->configValues[iCpuThreadNum]->GetUint64() != n_thd)
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. Your CPU config array has %llu members, while you want to use %llu threads.",
			int_port(n_thd), int_port(prv->configValues[iCpuThreadNum]->GetUint64()));
		return false;
	}

	thd_cfg c;
	for(size_t i=0; i < n_thd; i++)
	{
		if(!GetThreadConfig(i, c))
		{
			printer::inst()->print_msg(L0, "Thread %llu has invalid config.", int_port(i));
			return false;
		}
	}

	if(GetSlowMemSetting() == unknown_value)
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. use_slow_memory must be \"always\", \"no_mlck\", \"warn\" or \"never\"");
		return false;
	}

	if(!prv->configValues[iCallTimeout]->IsUint64() || !prv->configValues[iNetRetry]->IsUint64())
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. call_timeout and retry_time need to be positive integers.");
		return false;
	}

	if(!prv->configValues[iVerboseLevel]->IsUint64() || !prv->configValues[iAutohashTime]->IsUint64())
	{
		printer::inst()->print_msg(L0,
			"Invalid config file. verbose_level and h_print_time need to be positive integers.");
		return false;
	}

#ifdef _WIN32
	if(GetSlowMemSetting() == no_mlck)
	{
		printer::inst()->print_msg(L0, "On Windows large pages need mlock. Please use another option.\n");
		return false;
	}
#endif // _WIN32

	printer::inst()->set_verbose_level(prv->configValues[iVerboseLevel]->GetUint64());
	return true;
}
