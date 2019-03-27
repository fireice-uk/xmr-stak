
#include "xmrstak/backend/cpu/cpuType.hpp"

#include <cstdio>
#include <cstring>
#include <inttypes.h>

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
	std::memset(val, 0, sizeof(int32_t) * 4);

#ifdef _WIN32
	__cpuidex(val, eax, ecx);
#else
	__cpuid_count(eax, ecx, val[0], val[1], val[2], val[3]);
#endif
}

uint32_t get_masked(int32_t val, int32_t h, int32_t l)
{
	val &= (0x7FFFFFFF >> (31 - (h - l))) << l;
	return static_cast<uint32_t>(val >> l);
}

bool has_feature(int32_t val, int32_t bit)
{
	int32_t mask = 1 << bit;
	return (val & mask) != 0u;
}

Model getModel()
{
	Model result;

	int32_t cpu_HFP = 0;  // Highest Function Parameter
	int32_t cpu_HEFP = 0; // Highest Extended Function Parameter
	int32_t cpu_info[4];
	char cpustr[13] = {0};
	char brandstr[13] = {0};

	cpuid(0, 0, cpu_info);
	cpu_HFP = cpu_info[0];
	std::memcpy(cpustr, &cpu_info[1], 4);
	std::memcpy(cpustr + 4, &cpu_info[3], 4);
	std::memcpy(cpustr + 8, &cpu_info[2], 4);

	cpuid(1, 0, cpu_info);
	result.model = get_masked(cpu_info[0], 8, 4);
	result.family = get_masked(cpu_info[0], 12, 8);
	if(result.family == 0x6 || result.family == 0xF)
	{
		result.model += get_masked(cpu_info[0], 20, 16) << 4;
	}
	if(result.family != 0xF)
	{
		result.family += get_masked(cpu_info[0], 28, 20);
	}

	// feature bits https://en.wikipedia.org/wiki/CPUID#EAX=1:_Processor_Info_and_Feature_Bits
	// sse2/sse3/ssse3
	result.sse2 = has_feature(cpu_info[3], 26);
	result.sse3 = has_feature(cpu_info[2], 0);
	result.ssse3 = has_feature(cpu_info[2], 9);
	// aes-ni
	result.aes = has_feature(cpu_info[2], 25);
	// avx - 27 is the check if the OS overwrote cpu features
	result.avx = has_feature(cpu_info[2], 28) && has_feature(cpu_info[2], 27);

	// extended feature bits https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
	if(cpu_HFP >= 7)
	{
		cpuid(7, 0, cpu_info);
		result.avx2 = has_feature(cpu_info[1], 5);
	}
	// extended function support https://en.wikipedia.org/wiki/CPUID#EAX=80000000h:_Get_Highest_Extended_Function_Implemented
	cpuid(0x80000000, 0, cpu_info);
	cpu_HEFP = cpu_info[0];

	// processor brand string https://en.wikipedia.org/wiki/CPUID#EAX=80000002h,80000003h,80000004h:_Processor_Brand_String
	if(cpu_HEFP >= 0x80000004)
	{
		for(uint32_t efp=0x80000002; efp<0x80000004; efp++){
			cpuid(0x80000002, 0, cpu_info);
			std::memcpy(brandstr+(16*(efp-0x80000002)), &cpu_info, 16);
		}
		result.brand_name = brandstr;
	}

	if(strcmp(cpustr, "GenuineIntel") == 0)
	{
		if(result.family == 0x6){
			result.isIntelXBridge = (
				   result.model == 0x2A //Sandy Bridge
				|| result.model == 0x3A //Ivy Bridge
			);
			result.isIntelXWell = (
				   result.model == 0x3C || result.model == 0x45 || result.model == 0x46 //Haswell
				|| result.model == 0x47 || result.model == 0x3D //Broadwell
			);
			result.isIntelXLake = (
				   result.model == 0x4E || result.model == 0x5E //Skylake
				|| result.model == 0x8E //Kaby/Coffee/Whiskey/Amber Lake
				|| result.model == 0x9E //Kaby/Coffee Lake
				|| result.model == 0x66 //Cannon Lake
			);
		}
	}
	if(strcmp(cpustr, "AuthenticAMD") == 0)
	{
		result.isAMDHammer    = (result.family != 0x15 && result.family >= 0xF && result.family <= 0x16);
		result.isAMDBulldozer = (result.family == 0x15);
		result.isAMDZen       = (result.family == 0x17);
	}

	return result;
}

} // namespace cpu
} // namespace xmrstak
