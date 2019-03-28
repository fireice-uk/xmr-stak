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
#include "xmrstak/misc/console.hpp"
#include "xmrstak/misc/jext.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace xmrstak
{
namespace nvidia
{

using namespace rapidjson;

/*
 * This enum needs to match index in oConfigValues, otherwise we will get a runtime error
 */
enum configEnum
{
	aGpuThreadsConf
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
	{aGpuThreadsConf, "gpu_threads_conf", kNullType}};

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

constexpr size_t iConfigCnt = (sizeof(oConfigValues) / sizeof(oConfigValues[0]));

struct jconf::opaque_private
{
	Document jsonDoc;
	const Value* configValues[iConfigCnt]; //Compile time constant

	opaque_private()
	{
	}
};

bool jconf::NeedsAutoconf()
{
	return !prv->configValues[aGpuThreadsConf]->IsArray();
}

jconf* jconf::oInst = nullptr;

jconf::jconf()
{
	prv = new opaque_private();
}

size_t jconf::GetGPUThreadCount()
{
	if(prv->configValues[aGpuThreadsConf]->IsArray())
		return prv->configValues[aGpuThreadsConf]->Size();
	else
		return 0;
}

bool jconf::GetGPUThreadConfig(size_t id, thd_cfg& cfg)
{
	if(!prv->configValues[aGpuThreadsConf]->IsArray())
		return false;

	if(id >= prv->configValues[aGpuThreadsConf]->Size())
		return false;

	const Value& oThdConf = prv->configValues[aGpuThreadsConf]->GetArray()[id];

	if(!oThdConf.IsObject())
		return false;

	const Value *gid, *blocks, *threads, *bfactor, *bsleep, *aff, *syncMode, *memMode;
	gid = GetObjectMember(oThdConf, "index");
	blocks = GetObjectMember(oThdConf, "blocks");
	threads = GetObjectMember(oThdConf, "threads");
	bfactor = GetObjectMember(oThdConf, "bfactor");
	bsleep = GetObjectMember(oThdConf, "bsleep");
	aff = GetObjectMember(oThdConf, "affine_to_cpu");
	syncMode = GetObjectMember(oThdConf, "sync_mode");
	memMode = GetObjectMember(oThdConf, "mem_mode");

	if(gid == nullptr || blocks == nullptr || threads == nullptr ||
		bfactor == nullptr || bsleep == nullptr || aff == nullptr || syncMode == nullptr ||
		memMode == nullptr)
	{
		return false;
	}

	if(!gid->IsNumber() || gid->GetInt() < 0)
		return false;

	if(!blocks->IsNumber() || blocks->GetInt() < 0)
		return false;

	if(!threads->IsNumber() || threads->GetInt() < 0)
		return false;

	if(!bfactor->IsNumber() || bfactor->GetInt() < 0)
		return false;

	if(!bsleep->IsNumber() || bsleep->GetInt() < 0)
		return false;

	if(!aff->IsUint64() && !aff->IsBool())
		return false;

	if(!syncMode->IsNumber() || syncMode->GetInt() < 0 || syncMode->GetInt() > 3)
	{
		printer::inst()->print_msg(L0, "Error NVIDIA: sync_mode out of range or not a number. ( range: 0 <= sync_mode < 4.)");
		return false;
	}

	if(!memMode->IsNumber() || memMode->GetInt() < 0 || memMode->GetInt() > 1)
	{
		printer::inst()->print_msg(L0, "Error NVIDIA: mem_mode out of range or not a number. (range: 0 or 1)");
		return false;
	}

	cfg.id = gid->GetInt();
	cfg.blocks = blocks->GetInt();
	cfg.threads = threads->GetInt();
	cfg.bfactor = bfactor->GetInt();
	cfg.bsleep = bsleep->GetInt();
	cfg.syncMode = syncMode->GetInt();
	cfg.memMode = memMode->GetInt();

	if(aff->IsNumber())
		cfg.cpu_aff = aff->GetInt();
	else
		cfg.cpu_aff = -1;

	return true;
}

bool jconf::parse_config(const char* sFilename)
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

	prv->jsonDoc.Parse<kParseCommentsFlag | kParseTrailingCommasFlag>(buffer, flen + 2);
	free(buffer);

	if(prv->jsonDoc.HasParseError())
	{
		printer::inst()->print_msg(L0, "JSON config parse error in '%s' (offset %llu): %s",
			sFilename, int_port(prv->jsonDoc.GetErrorOffset()), GetParseError_En(prv->jsonDoc.GetParseError()));
		return false;
	}

	if(!prv->jsonDoc.IsObject())
	{ //This should never happen as we created the root ourselves
		printer::inst()->print_msg(L0, "Invalid config file '%s'. No root?", sFilename);
		return false;
	}

	for(size_t i = 0; i < iConfigCnt; i++)
	{
		if(oConfigValues[i].iName != i)
		{
			printer::inst()->print_msg(L0, "Code error. oConfigValues are not in order. %s", oConfigValues[i].sName);
			return false;
		}

		prv->configValues[i] = GetObjectMember(prv->jsonDoc, oConfigValues[i].sName);

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

	return true;
}

} // namespace nvidia
} // namespace xmrstak
