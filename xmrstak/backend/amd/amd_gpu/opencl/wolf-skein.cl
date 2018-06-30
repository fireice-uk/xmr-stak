R"===(
#ifndef WOLF_SKEIN_CL
#define WOLF_SKEIN_CL

// Vectorized Skein implementation macros and functions by Wolf
// Updated by taisel

#define SKEIN_KS_PARITY	0x1BD11BDAA9FC1A22

static const __constant ulong SKEIN256_IV[8] =
{
	0xCCD044A12FDB3E13UL, 0xE83590301A79A9EBUL,
	0x55AEA0614F816E6FUL, 0x2A2767A4AE9B94DBUL,
	0xEC06025E74DD7683UL, 0xE7A436CDC4746251UL,
	0xC36FBAF9393AD185UL, 0x3EEDBA1833EDFC13UL
};

static const __constant ulong SKEIN512_256_IV[8] =
{
	0xCCD044A12FDB3E13UL, 0xE83590301A79A9EBUL,
	0x55AEA0614F816E6FUL, 0x2A2767A4AE9B94DBUL,
	0xEC06025E74DD7683UL, 0xE7A436CDC4746251UL,
	0xC36FBAF9393AD185UL, 0x3EEDBA1833EDFC13UL
};

#define SKEIN_INJECT_KEY(p, s, q)	do { \
	p += h; \
	p.s5 += t[s]; \
	p.s6 += t[select(s + 1U, 0U, s == 2U)]; \
	p.s7 += q; \
} while(0)

ulong SKEIN_ROT(const uint2 x, const uint y)
{
	if(y < 32) return(as_ulong(amd_bitalign(x, x.s10, 32 - y)));
	else return(as_ulong(amd_bitalign(x.s10, x, 32 - (y - 32))));
}

void SkeinMix8(ulong4 *pv0, ulong4 *pv1, const ulong4 rc)
{
	*pv0 += *pv1;
	(*pv1).s0 = SKEIN_ROT(as_uint2((*pv1).s0), rc.s0);
	(*pv1).s1 = SKEIN_ROT(as_uint2((*pv1).s1), rc.s1);
	(*pv1).s2 = SKEIN_ROT(as_uint2((*pv1).s2), rc.s2);
	(*pv1).s3 = SKEIN_ROT(as_uint2((*pv1).s3), rc.s3);
	*pv1 ^= *pv0;
}

ulong8 SkeinEvenRound(ulong8 p, const ulong8 h, const ulong *t, const uint s, const uint q)
{
	SKEIN_INJECT_KEY(p, s, q);
	ulong4 pv0 = p.even, pv1 = p.odd;

	SkeinMix8(&pv0, &pv1, (ulong4)(46, 36, 19, 37));
	pv0 = shuffle(pv0, (ulong4)(1, 2, 3, 0));
	pv1 = shuffle(pv1, (ulong4)(0, 3, 2, 1));

	SkeinMix8(&pv0, &pv1, (ulong4)(33, 27, 14, 42));
	pv0 = shuffle(pv0, (ulong4)(1, 2, 3, 0));
	pv1 = shuffle(pv1, (ulong4)(0, 3, 2, 1));

	SkeinMix8(&pv0, &pv1, (ulong4)(17, 49, 36, 39));
	pv0 = shuffle(pv0, (ulong4)(1, 2, 3, 0));
	pv1 = shuffle(pv1, (ulong4)(0, 3, 2, 1));

	SkeinMix8(&pv0, &pv1, (ulong4)(44, 9, 54, 56));
	return(shuffle2(pv0, pv1, (ulong8)(1, 4, 2, 7, 3, 6, 0, 5)));
}

ulong8 SkeinOddRound(ulong8 p, const ulong8 h, const ulong *t, const uint s, const uint q)
{
	SKEIN_INJECT_KEY(p, s, q);
    ulong4 pv0 = p.even, pv1 = p.odd;

	SkeinMix8(&pv0, &pv1, (ulong4)(39, 30, 34, 24));
	pv0 = shuffle(pv0, (ulong4)(1, 2, 3, 0));
	pv1 = shuffle(pv1, (ulong4)(0, 3, 2, 1));

	SkeinMix8(&pv0, &pv1, (ulong4)(13, 50, 10, 17));
	pv0 = shuffle(pv0, (ulong4)(1, 2, 3, 0));
	pv1 = shuffle(pv1, (ulong4)(0, 3, 2, 1));

	SkeinMix8(&pv0, &pv1, (ulong4)(25, 29, 39, 43));
	pv0 = shuffle(pv0, (ulong4)(1, 2, 3, 0));
	pv1 = shuffle(pv1, (ulong4)(0, 3, 2, 1));

	SkeinMix8(&pv0, &pv1, (ulong4)(8, 35, 56, 22));
	return(shuffle2(pv0, pv1, (ulong8)(1, 4, 2, 7, 3, 6, 0, 5)));
}

ulong8 Skein512Block(ulong8 p, ulong8 h, ulong h8, const ulong *t)
{
	#pragma unroll
	for(int i = 0; i < 18; ++i)
	{
		p = SkeinEvenRound(p, h, t, 0U, i);
		++i;
		ulong tmp = h.s0;
		h = shuffle(h, (ulong8)(1, 2, 3, 4, 5, 6, 7, 0));
		h.s7 = h8;
		h8 = tmp;
		p = SkeinOddRound(p, h, t, 1U, i);
		++i;
		tmp = h.s0;
		h = shuffle(h, (ulong8)(1, 2, 3, 4, 5, 6, 7, 0));
		h.s7 = h8;
		h8 = tmp;
		p = SkeinEvenRound(p, h, t, 2U, i);
		++i;
		tmp = h.s0;
		h = shuffle(h, (ulong8)(1, 2, 3, 4, 5, 6, 7, 0));
		h.s7 = h8;
		h8 = tmp;
		p = SkeinOddRound(p, h, t, 0U, i);
		++i;
		tmp = h.s0;
		h = shuffle(h, (ulong8)(1, 2, 3, 4, 5, 6, 7, 0));
		h.s7 = h8;
		h8 = tmp;
		p = SkeinEvenRound(p, h, t, 1U, i);
		++i;
		tmp = h.s0;
		h = shuffle(h, (ulong8)(1, 2, 3, 4, 5, 6, 7, 0));
		h.s7 = h8;
		h8 = tmp;
		p = SkeinOddRound(p, h, t, 2U, i);
		tmp = h.s0;
		h = shuffle(h, (ulong8)(1, 2, 3, 4, 5, 6, 7, 0));
		h.s7 = h8;
		h8 = tmp;
	}

	p += h;
	p.s5 += t[0];
	p.s6 += t[1];
	p.s7 += 18;
	return(p);
}

#endif

)==="
