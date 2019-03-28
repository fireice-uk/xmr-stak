#pragma once

#include "xmrstak/params.hpp"

#include <stdlib.h>
#include <string>

namespace xmrstak
{
namespace amd
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

	bool parse_config(const char* sFilename = params::inst().configFileAMD.c_str());

	struct thd_cfg
	{
		size_t index;
		size_t intensity;
		size_t w_size;
		long long cpu_aff;
		int stridedIndex;
		int interleave = 40;
		int memChunk;
		int unroll;
		bool compMode;
	};

	size_t GetThreadCount();
	bool GetThreadConfig(size_t id, thd_cfg& cfg);

	size_t GetAutoTune();
	size_t GetPlatformIdx();

  private:
	jconf();
	static jconf* oInst;

	struct opaque_private;
	opaque_private* prv;
};

} // namespace amd
} // namespace xmrstak
