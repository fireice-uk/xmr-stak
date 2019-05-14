#pragma once

#include "xmrstak/backend/cryptonight.hpp"
#include <stdint.h>

#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#include <malloc.h>
#define HAS_WIN_INTRIN_API
#endif

#ifdef __GNUC__
#include <x86intrin.h>
#if !defined(HAS_WIN_INTRIN_API)
#include <cpuid.h>
#endif // !defined(HAS_WIN_INTRIN_API)
#endif // __GNUC__

inline void cngpu_cpuid(uint32_t eax, int32_t ecx, int32_t val[4])
{
	val[0] = 0;
	val[1] = 0;
	val[2] = 0;
	val[3] = 0;

#if defined(HAS_WIN_INTRIN_API)
	__cpuidex(val, eax, ecx);
#else
	__cpuid_count(eax, ecx, val[0], val[1], val[2], val[3]);
#endif
}

inline bool cngpu_check_avx2()
{
	int32_t cpu_info[4];
	cngpu_cpuid(7, 0, cpu_info);
	return (cpu_info[1] & (1 << 5)) != 0;
}

void cn_gpu_inner_avx(const uint8_t* spad, uint8_t* lpad, const xmrstak_algo& algo);

void cn_gpu_inner_ssse3(const uint8_t* spad, uint8_t* lpad, const xmrstak_algo& algo);
