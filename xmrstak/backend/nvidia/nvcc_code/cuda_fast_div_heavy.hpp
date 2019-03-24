#pragma once

#include <stdint.h>

__device__ __forceinline__ int64_t fast_div_heavy(int64_t _a, int _b)
{

	uint64_t a = abs(_a);
	int b = abs(_b);

	float rcp = __frcp_rn(__int2float_rn(b));
	float rcp2 = __uint_as_float(__float_as_uint(rcp) + (32U << 23));

	uint64_t q1 = __float2ull_rz(__int2float_rn(((int*)&a)[1]) * rcp2);
	a -= q1 * static_cast<uint32_t>(b);

	uint64_t tmp = a >> 12;
	float q2f = __int2float_rn(((int*)&tmp)[0]) * rcp;
	q2f = __uint_as_float(__float_as_uint(q2f) + (12U << 23));
	int64_t q2 = __float2ll_rn(q2f);
	int a2 = ((int*)&a)[0] - ((int*)&q2)[0] * b;

	int q3 = __float2int_rn(__int2float_rn(a2) * rcp);
	q3 += (a2 - q3 * b) >> 31;

	const uint64_t q = q1 + q2 + q3;
	return ((((int*)&_a)[1] ^ _b) < 0) ? -q : q;
}
