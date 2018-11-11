R"===(
#ifndef FAST_DIV_HEAVY_CL
#define FAST_DIV_HEAVY_CL

inline ulong get_reciprocal_heavy(uint a)
{
	const uint shift = clz(a);
	a <<= shift;

	const float a_hi = as_float((a >> 8) + 1 + ((126U + 31U) << 23));
	const float a_lo = convert_float_rte(as_int(a & 0xFF) - 256);

	const float r = native_recip(a_hi);

	const uint tmp0 = as_uint(r);
	const uint tmp1 = tmp0 + ((shift + 2 + 64U) << 23);
	const float r_scaled = as_float(tmp1);

	const float h = fma(a_lo, r, fma(a_hi, r, -1.0f));

	const float r_scaled_hi = as_float(tmp1 & ~4095U);
	const float h_hi = as_float(as_uint(h) & ~4095U);

	const float r_scaled_lo = r_scaled - r_scaled_hi;
	const float h_lo = h - h_hi;

	const float x1 = h_hi * r_scaled_hi;
	const float x2 = h_lo * r_scaled + h_hi * r_scaled_lo;

	const long h1 = convert_long_rte(x1);
	const int h2 = convert_int_rtp(x2) - convert_int_rtn(h * (x1 + x2));

	const ulong result = tmp0 & 0xFFFFFF;
	return (result << (shift + 9)) - ((h1 + h2) >> 2);
}

inline long fast_div_heavy(long _a, int _b)
{
	const ulong a = abs(_a);
	const uint b = abs(_b);
	ulong q = mul_hi(a, get_reciprocal_heavy(b));

	const long tmp = a - q * b;
	const int overshoot = (tmp < 0) ? 1 : 0;
	const int undershoot = (tmp >= b) ? 1 : 0;
	q += undershoot - overshoot;

	return ((as_int2(_a).s1 ^ _b) < 0) ? -q : q;
}

#endif
)==="
        