#pragma once

#include <stdint.h>

static __constant__ const uint32_t RCP_C[256] =
{
	0xfe01be73u,0xfd07ff01u,0xfa118c5au,0xf924fb13u,0xf630cddbu,0xf558f73cu,0xf25f2934u,0xf1a3f37bu,
	0xee9c4562u,0xee02efd0u,0xeae7ced5u,0xea76ec3au,0xe7417330u,0xe6ffe8b8u,0xe3a8e217u,0xe39be54au,
	0xe01dcd03u,0xe04ae1f0u,0xdc9fea3bu,0xdd0bdea8u,0xd92eef38u,0xd9dedb73u,0xd5ca9626u,0xd6c3d84fu,
	0xd27299dcu,0xd3b9d53cu,0xcf26b659u,0xd0bfd23au,0xcbe6ab09u,0xcdd5cf48u,0xc8b23886u,0xcafacc65u,
	0xc58920e5u,0xc82ec992u,0xc26b283eu,0xc572c6ceu,0xbf5813d7u,0xc2c3c419u,0xbc4facdbu,0xc023c171u,
	0xb951b9f6u,0xbd8fbed7u,0xb65e05c8u,0xbb09bc4bu,0xb3745d97u,0xb890b9cbu,0xb0948d04u,0xb624b758u,
	0xadbe61e8u,0xb3c3b4f2u,0xaaf1ae2au,0xb16eb297u,0xa82e412eu,0xaf25b048u,0xa573ec98u,0xace7ae05u,
	0xa2c28519u,0xaab4abcdu,0xa019df1cu,0xa88ca99fu,0x9d79cf91u,0xa66ea77cu,0x9ae22df8u,0xa45ba563u,
	0x9852d0ceu,0xa251a354u,0x95cb912eu,0xa050a14fu,0x934c48d6u,0x9e5a9f54u,0x90d4d228u,0x9c6c9d62u,
	0x8e650939u,0x9a879b79u,0x8bfccaf5u,0x98ac9998u,0x899bf212u,0x96d897c1u,0x87425eedu,0x950d95f2u,
	0x84efefd3u,0x934a942bu,0x82a48450u,0x918f926cu,0x805ffcb4u,0x8fdc90b5u,0x7e223ab7u,0x8e308f05u,
	0x7beb1f71u,0x8c8c8d5du,0x79ba8ce2u,0x8aef8bbdu,0x7790683eu,0x89598a23u,0x756c9343u,0x87ca8891u,
	0x734ef468u,0x86428705u,0x71376efbu,0x84c18581u,0x6f25e9ebu,0x83458402u,0x6d1a4b34u,0x81d0828au,
	0x6b147a52u,0x80628118u,0x69145cfbu,0x7ef97fadu,0x6719dd39u,0x7d967e47u,0x6524e2abu,0x7c397ce7u,
	0x6335561bu,0x7ae27b8du,0x614b21eau,0x79907a38u,0x5f662f10u,0x784478e9u,0x5d8667dfu,0x76fd77a0u,
	0x5babb887u,0x75bb765bu,0x59d60b2eu,0x747e751cu,0x58054d25u,0x734673e1u,0x5639688fu,0x721372acu,
	0x54724c2du,0x70e5717bu,0x52afe29cu,0x6fbb7050u,0x50f21c05u,0x6e966f28u,0x4f38e412u,0x6d766e06u,
	0x4d842a91u,0x6c5a6ce7u,0x4bd3dcd0u,0x6b426bcdu,0x4a27e96au,0x6a2e6ab8u,0x4880415eu,0x691f69a6u,
	0x46dcd25du,0x68136899u,0x453d8df4u,0x670c678fu,0x43a262a5u,0x6608668au,0x420b42d6u,0x65096588u,
	0x40781dd3u,0x640d648au,0x3ee8e49au,0x63146390u,0x3d5d8a11u,0x621f6299u,0x3bd5fee0u,0x612e61a6u,
	0x3a523496u,0x604060b7u,0x38d21e75u,0x5f565fcbu,0x3755aec4u,0x5e6f5ee2u,0x35dcd78fu,0x5d8b5dfdu,
	0x34678d72u,0x5cab5d1au,0x32f5c17cu,0x5bcd5c3bu,0x318767f1u,0x5af35b60u,0x301c7511u,0x5a1b5a87u,
	0x2eb4dccau,0x594759b1u,0x2d50935cu,0x587658deu,0x2bef8bfau,0x57a7580eu,0x2a91bc5cu,0x56db5741u,
	0x2937198fu,0x56125676u,0x27df970eu,0x554c55afu,0x268b2b78u,0x548854eau,0x2539cba1u,0x53c75428u,
	0x23eb6d84u,0x53095368u,0x22a00644u,0x524d52abu,0x21578cd3u,0x519451f0u,0x2011f5f9u,0x50dd5138u,
	0x1ecf388eu,0x50285082u,0x1d8f4b53u,0x4f764fcfu,0x1c5224abu,0x4ec64f1eu,0x1b17bb87u,0x4e184e6fu,
	0x19e0073fu,0x4d6d4dc2u,0x18aafe0au,0x4cc44d18u,0x177896f3u,0x4c1c4c70u,0x1648cb16u,0x4b784bcau,
	0x151b9051u,0x4ad54b26u,0x13f0deeau,0x4a344a84u,0x12c8aef3u,0x499549e4u,0x11a2f829u,0x48f84946u,
	0x107fb1ffu,0x485d48abu,0xf5ed5f0u,0x47c44811u,0xe405bc1u,0x472d4779u,0xd243bdau,0x469846e3u,
	0xc0a6fa1u,0x4605464eu,0xaf2edf2u,0x457345bcu,0x9ddb163u,0x44e3452bu,0x8cab264u,0x4455449cu,
	0x7b9e9d5u,0x43c9440fu,0x6ab5173u,0x433e4383u,0x59ee141u,0x42b542fau,0x49494c7u,0x422e4271u,
	0x38c62ffu,0x41a841ebu,0x286478bu,0x41244166u,0x1823b84u,0x40a140e2u,0x803883u,0x401C4060u,
};

__device__ __forceinline__ uint32_t get_reciprocal(const uint32_t* RCP, uint32_t a)
{
	const uint32_t index1 = (a & 0x7F000000U) >> 23;
	const int index2 = (int)((a >> 8) & 0xFFFFU) - 32768;

	const uint32_t r1 = RCP[index1];
	uint32_t r2_0 = RCP[index1 + 1];
	if (index2 > 0) r2_0 >>= 16;
	const int r2 = r2_0 & 0xFFFFU;

	const uint32_t r = r1 - (uint32_t)(__mul24(r2, index2) >> 6);

	const uint64_t lo0 = (uint64_t)(r) * a;
	uint64_t lo = lo0 + ((uint64_t)(a) << 32);

	a >>= 1;
	const bool b = (a >= lo) || (lo >= lo0);
	lo = a - lo;

	const uint64_t k = __umulhi((uint32_t)lo, r) + ((uint64_t)(r) * ((uint32_t*)&lo)[1]) + lo;
	return ((uint32_t*)&k)[1] + (b ? r : 0);
}

__device__ __forceinline__ uint64_t fast_div_v2(const uint32_t *RCP, uint64_t a, uint32_t b)
{
	const uint32_t r = get_reciprocal(RCP, b);
	const uint64_t k = __umulhi((uint32_t)a, r) + ((uint64_t)(r) * ((uint32_t*)&a)[1]) + a;

	uint32_t q[2];
	q[0] = ((uint32_t*)&k)[1];
	q[1] = (k < a) ? 1 : 0;

	const int64_t tmp = a - *((uint64_t*)(q)) * b;
	const uint32_t overshoot = (tmp < 0) ? 1u : 0U;
	const uint32_t undershoot = (tmp >= b) ? 1u : 0U;

	q[0] += undershoot - overshoot;
	q[1] = (uint32_t)(tmp) + (overshoot == 1 ? b : 0U) - (undershoot ? b : 0U);

	return *((uint64_t*)(q));
}

__device__ __forceinline__ uint32_t fast_sqrt_v2(const uint64_t n1)
{
	float x = __uint_as_float((((uint32_t*)&n1)[1] >> 9) + ((64U + 127U) << 23));
	float x1;
	asm("rsqrt.approx.f32 %0, %1;" : "=f"(x1) : "f"(x));
	asm("sqrt.approx.f32 %0, %1;" : "=f"(x) : "f"(x));

	// The following line does x1 *= 4294967296.0f;
	x1 = __uint_as_float(__float_as_uint(x1) + (32U << 23));

	const uint32_t x0 = __float_as_uint(x) - (158U << 23);
	const int64_t delta0 = n1 - (((int64_t)(x0) * x0) << 18);
	const float delta = __int2float_rn(((int32_t*)&delta0)[1]) * x1;

	uint32_t result = (x0 << 10) + __float2int_rn(delta);
	const uint32_t s = result >> 1;
	const uint32_t b = result & 1;

	const uint64_t x2 = (uint64_t)(s) * (s + b) + ((uint64_t)(result) << 32) - n1;
	const int32_t overshoot = ((int64_t)(x2 + b) > 0) ? -1 : 0;
	const int32_t undershoot = ((int64_t)(x2 + 0x100000000UL + s) < 0) ? 1 : 0;
	result += (overshoot+undershoot);

	return result;
}
