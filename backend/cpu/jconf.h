#pragma once
#include <stdlib.h>
#include <string>

namespace xmrstak
{
namespace cpu
{

class jconf
{
public:
	static jconf* inst()
	{
		if (oInst == nullptr) oInst = new jconf;
		return oInst;
	};

	bool parse_config(const char* sFilename = "cpu.txt");

	struct thd_cfg {
		bool bDoubleMode;
		bool bNoPrefetch;
		long long iCpuAff;
	};

	size_t GetThreadCount();
	bool GetThreadConfig(size_t id, thd_cfg &cfg);
	bool NeedsAutoconf();

	


private:
	jconf();
	static jconf* oInst;

	struct opaque_private;
	opaque_private* prv;
};

} // namespace cpu
} // namepsace xmrstak
