#pragma once
#include "xmrstak/params.hpp"
#include <stdlib.h>
#include <string>

namespace xmrstak
{
namespace nvidia
{

class jconf
{
  public:
	static jconf* inst()
	{
		if(oInst == nullptr)
			oInst = new jconf;
		return oInst;
	};

	bool parse_config(const char* sFilename = params::inst().configFileNVIDIA.c_str());

	struct thd_cfg
	{
		uint32_t id;
		uint32_t blocks;
		uint32_t threads;
		uint32_t bfactor;
		uint32_t bsleep;
		bool bDoubleMode;
		bool bNoPrefetch;
		int32_t cpu_aff;
		int syncMode;
		int memMode;

		long long iCpuAff;
	};

	size_t GetGPUThreadCount();

	bool GetGPUThreadConfig(size_t id, thd_cfg& cfg);

	bool NeedsAutoconf();

  private:
	jconf();
	static jconf* oInst;

	struct opaque_private;
	opaque_private* prv;
};

} // namespace nvidia
} // namespace xmrstak
