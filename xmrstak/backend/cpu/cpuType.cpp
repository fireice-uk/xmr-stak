
#include "xmrstak/backend/cpu/cpuType.hpp"

#include <cstring>
#include <inttypes.h>
#include <cstdio>

#ifdef _WIN32
#define strcasecmp _stricmp
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace xmrstak
{
namespace cpu
{
	void cpuid(uint32_t eax, int32_t ecx, int32_t val[4])
	{
		std::memset(val, 0, sizeof(int32_t)*4);

	#ifdef _WIN32
		__cpuidex(val, eax, ecx);
	#else
		__cpuid_count(eax, ecx, val[0], val[1], val[2], val[3]);
	#endif
	}

	int32_t get_masked(int32_t val, int32_t h, int32_t l)
	{
		val &= (0x7FFFFFFF >> (31-(h-l))) << l;
		return val >> l;
	}

	bool has_feature(int32_t val, int32_t bit)
	{
		int32_t mask = 1 << bit;
		return (val & mask) != 0u;
		
	}
	
	Model getModel()
	{
		int32_t cpu_info[4];
		char cpustr[13] = {0};

		cpuid(0, 0, cpu_info);
		std::memcpy(cpustr, &cpu_info[1], 4);
		std::memcpy(cpustr+4, &cpu_info[3], 4);
		std::memcpy(cpustr+8, &cpu_info[2], 4);

		Model result;

		cpuid(1, 0, cpu_info);
		
		result.family = get_masked(cpu_info[0], 12, 8);
		result.model = get_masked(cpu_info[0], 8, 4) | get_masked(cpu_info[0], 20, 16) << 4;
		result.type_name = cpustr;

		// feature bits https://en.wikipedia.org/wiki/CPUID
		// sse2
		result.sse2 = has_feature(cpu_info[3], 26);
		// aes-ni
		result.aes = has_feature(cpu_info[2], 25);
		// avx
		result.avx = has_feature(cpu_info[2], 28);	

		if(strcmp(cpustr, "AuthenticAMD") == 0)
		{
			if(result.family == 0xF)
				result.family += get_masked(cpu_info[0], 28, 20);
		}

		return result;
	}

} // namespace cpu
} // namespace xmrstak
