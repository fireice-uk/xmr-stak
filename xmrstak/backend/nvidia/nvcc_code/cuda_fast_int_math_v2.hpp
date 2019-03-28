#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t get_reciprocal(uint32_t a)
{
	const float a_hi = __uint_as_float((a >> 8) + ((126U + 31U) << 23));
	const float a_lo = __uint2float_rn(a & 0xFF);

	float r = __frcp_rn(a_hi);
	const float r_scaled = __uint_as_float(__float_as_uint(r) + (64U << 23));

	const float h = __fmaf_rn(a_lo, r, __fmaf_rn(a_hi, r, -1.0f));
	return (__float_as_uint(r) << 9) - __float2int_rn(h * r_scaled);
}

__device__ __forceinline__ uint64_t fast_div_v2(uint64_t a, uint32_t b)
{
	const uint32_t r = get_reciprocal(b);
	const uint32_t a1 = ((uint32_t*)&a)[1];
	const uint64_t k = __umulhi(((uint32_t*)&a)[0], r) + ((uint64_t)(r)*a1) + a;

	const uint32_t q = ((uint32_t*)&k)[1];
	int64_t tmp = a - ((uint64_t)(q)*b);
	((int32_t*)(&tmp))[1] -= q < a1 ? b : 0;

	const int overshoot = ((int*)(&tmp))[1] >> 31;
	const int64_t tmp_u = (uint32_t)(b - 1) - tmp;
	const int undershoot = ((int*)&tmp_u)[1] >> 31;

	uint64_t result;
	((uint32_t*)&result)[0] = q + overshoot - undershoot;
	((uint32_t*)&result)[1] = ((uint32_t*)(&tmp))[0] + ((uint32_t)(overshoot)&b) - ((uint32_t)(undershoot)&b);

	return result;
}

__device__ __forceinline__ uint32_t fast_sqrt_v2(const uint64_t n1)
{
	float x = __uint_as_float((((uint32_t*)&n1)[1] >> 9) + ((64U + 127U) << 23));
	float x1;
	asm("rsqrt.approx.f32 %0, %1;"
		: "=f"(x1)
		: "f"(x));
	asm("sqrt.approx.f32 %0, %1;"
		: "=f"(x)
		: "f"(x));

	// The following line does x1 *= 4294967296.0f;
	x1 = __uint_as_float(__float_as_uint(x1) + (32U << 23));

	const uint32_t x0 = __float_as_uint(x) - (158U << 23);
	const int64_t delta0 = n1 - (((int64_t)(x0)*x0) << 18);
	const float delta = __int2float_rn(((int32_t*)&delta0)[1]) * x1;

	uint32_t result = (x0 << 10) + __float2int_rn(delta);
	const uint32_t s = result >> 1;
	const uint32_t b = result & 1;

	const uint64_t x2 = (uint64_t)(s) * (s + b) + ((uint64_t)(result) << 32) - n1;
	const int32_t overshoot = ((int64_t)(x2 + b) > 0) ? -1 : 0;
	const int32_t undershoot = ((int64_t)(x2 + 0x100000000UL + s) < 0) ? 1 : 0;
	result += (overshoot + undershoot);
	return result;
}
