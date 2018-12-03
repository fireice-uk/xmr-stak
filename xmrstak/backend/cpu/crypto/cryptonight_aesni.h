/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  *
  */
#pragma once

#include "cryptonight.h"
#include "xmrstak/backend/cryptonight.hpp"
#include <memory.h>
#include <stdio.h>
#include <cfenv>
#include <utility>

#ifdef __GNUC__
#include <x86intrin.h>
static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t* hi)
{
	unsigned __int128 r = (unsigned __int128)a * (unsigned __int128)b;
	*hi = r >> 64;
	return (uint64_t)r;
}

#else
#include <intrin.h>
#endif // __GNUC__

#if !defined(_LP64) && !defined(_WIN64)
#error You are trying to do a 32-bit build. This will all end in tears. I know it.
#endif

#include "soft_aes.hpp"

extern "C"
{
	void keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen);
	void keccakf(uint64_t st[25], int rounds);
	extern void(*const extra_hashes[4])(const void *, uint32_t, char *);
}

// This will shift and xor tmp1 into itself as 4 32-bit vals such as
// sl_xor(a1 a2 a3 a4) = a1 (a2^a1) (a3^a2^a1) (a4^a3^a2^a1)
static inline __m128i sl_xor(__m128i tmp1)
{
	__m128i tmp4;
	tmp4 = _mm_slli_si128(tmp1, 0x04);
	tmp1 = _mm_xor_si128(tmp1, tmp4);
	tmp4 = _mm_slli_si128(tmp4, 0x04);
	tmp1 = _mm_xor_si128(tmp1, tmp4);
	tmp4 = _mm_slli_si128(tmp4, 0x04);
	tmp1 = _mm_xor_si128(tmp1, tmp4);
	return tmp1;
}

template<uint8_t rcon>
static inline void aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
	__m128i xout1 = _mm_aeskeygenassist_si128(*xout2, rcon);
	xout1 = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
	*xout0 = sl_xor(*xout0);
	*xout0 = _mm_xor_si128(*xout0, xout1);
	xout1 = _mm_aeskeygenassist_si128(*xout0, 0x00);
	xout1 = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
	*xout2 = sl_xor(*xout2);
	*xout2 = _mm_xor_si128(*xout2, xout1);
}

static inline void soft_aes_genkey_sub(__m128i* xout0, __m128i* xout2, uint8_t rcon)
{
	__m128i xout1 = soft_aeskeygenassist(*xout2, rcon);
	xout1 = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
	*xout0 = sl_xor(*xout0);
	*xout0 = _mm_xor_si128(*xout0, xout1);
	xout1 = soft_aeskeygenassist(*xout0, 0x00);
	xout1 = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
	*xout2 = sl_xor(*xout2);
	*xout2 = _mm_xor_si128(*xout2, xout1);
}

template<bool SOFT_AES>
static inline void aes_genkey(const __m128i* memory, __m128i* k0, __m128i* k1, __m128i* k2, __m128i* k3,
	__m128i* k4, __m128i* k5, __m128i* k6, __m128i* k7, __m128i* k8, __m128i* k9)
{
	__m128i xout0, xout2;

	xout0 = _mm_load_si128(memory);
	xout2 = _mm_load_si128(memory+1);
	*k0 = xout0;
	*k1 = xout2;

	if(SOFT_AES)
		soft_aes_genkey_sub(&xout0, &xout2, 0x01);
	else
		aes_genkey_sub<0x01>(&xout0, &xout2);
	*k2 = xout0;
	*k3 = xout2;

	if(SOFT_AES)
		soft_aes_genkey_sub(&xout0, &xout2, 0x02);
	else
		aes_genkey_sub<0x02>(&xout0, &xout2);
	*k4 = xout0;
	*k5 = xout2;

	if(SOFT_AES)
		soft_aes_genkey_sub(&xout0, &xout2, 0x04);
	else
		aes_genkey_sub<0x04>(&xout0, &xout2);
	*k6 = xout0;
	*k7 = xout2;

	if(SOFT_AES)
		soft_aes_genkey_sub(&xout0, &xout2, 0x08);
	else
		aes_genkey_sub<0x08>(&xout0, &xout2);
	*k8 = xout0;
	*k9 = xout2;
}

static inline void aes_round(__m128i key, __m128i* x0, __m128i* x1, __m128i* x2, __m128i* x3, __m128i* x4, __m128i* x5, __m128i* x6, __m128i* x7)
{
	*x0 = _mm_aesenc_si128(*x0, key);
	*x1 = _mm_aesenc_si128(*x1, key);
	*x2 = _mm_aesenc_si128(*x2, key);
	*x3 = _mm_aesenc_si128(*x3, key);
	*x4 = _mm_aesenc_si128(*x4, key);
	*x5 = _mm_aesenc_si128(*x5, key);
	*x6 = _mm_aesenc_si128(*x6, key);
	*x7 = _mm_aesenc_si128(*x7, key);
}

static inline void soft_aes_round(__m128i key, __m128i* x0, __m128i* x1, __m128i* x2, __m128i* x3, __m128i* x4, __m128i* x5, __m128i* x6, __m128i* x7)
{
	*x0 = soft_aesenc(*x0, key);
	*x1 = soft_aesenc(*x1, key);
	*x2 = soft_aesenc(*x2, key);
	*x3 = soft_aesenc(*x3, key);
	*x4 = soft_aesenc(*x4, key);
	*x5 = soft_aesenc(*x5, key);
	*x6 = soft_aesenc(*x6, key);
	*x7 = soft_aesenc(*x7, key);
}

inline void mix_and_propagate(__m128i& x0, __m128i& x1, __m128i& x2, __m128i& x3, __m128i& x4, __m128i& x5, __m128i& x6, __m128i& x7)
{
	__m128i tmp0 = x0;
	x0 = _mm_xor_si128(x0, x1);
	x1 = _mm_xor_si128(x1, x2);
	x2 = _mm_xor_si128(x2, x3);
	x3 = _mm_xor_si128(x3, x4);
	x4 = _mm_xor_si128(x4, x5);
	x5 = _mm_xor_si128(x5, x6);
	x6 = _mm_xor_si128(x6, x7);
	x7 = _mm_xor_si128(x7, tmp0);
}

template<size_t MEM, bool SOFT_AES, bool PREFETCH, xmrstak_algo ALGO>
void cn_explode_scratchpad(const __m128i* input, __m128i* output)
{
	// This is more than we have registers, compiler will assign 2 keys on the stack
	__m128i xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7;
	__m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

	aes_genkey<SOFT_AES>(input, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

	xin0 = _mm_load_si128(input + 4);
	xin1 = _mm_load_si128(input + 5);
	xin2 = _mm_load_si128(input + 6);
	xin3 = _mm_load_si128(input + 7);
	xin4 = _mm_load_si128(input + 8);
	xin5 = _mm_load_si128(input + 9);
	xin6 = _mm_load_si128(input + 10);
	xin7 = _mm_load_si128(input + 11);

	if(ALGO == cryptonight_heavy || ALGO == cryptonight_haven || ALGO == cryptonight_bittube2)
	{
		for(size_t i=0; i < 16; i++)
		{
			if(SOFT_AES)
			{
				soft_aes_round(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				soft_aes_round(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			}
			else
			{
				aes_round(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
				aes_round(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			}
			mix_and_propagate(xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
		}
	}

	for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8)
	{
		if(SOFT_AES)
		{
			soft_aes_round(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			soft_aes_round(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
		}
		else
		{
			aes_round(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
			aes_round(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
		}

		_mm_store_si128(output + i + 0, xin0);
		_mm_store_si128(output + i + 1, xin1);
		_mm_store_si128(output + i + 2, xin2);
		_mm_store_si128(output + i + 3, xin3);

		if(PREFETCH)
			_mm_prefetch((const char*)output + i + 0, _MM_HINT_T2);

		_mm_store_si128(output + i + 4, xin4);
		_mm_store_si128(output + i + 5, xin5);
		_mm_store_si128(output + i + 6, xin6);
		_mm_store_si128(output + i + 7, xin7);

		if(PREFETCH)
			_mm_prefetch((const char*)output + i + 4, _MM_HINT_T2);
	}
}

template<size_t MEM, bool SOFT_AES, bool PREFETCH, xmrstak_algo ALGO>
void cn_implode_scratchpad(const __m128i* input, __m128i* output)
{
	// This is more than we have registers, compiler will assign 2 keys on the stack
	__m128i xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7;
	__m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

	aes_genkey<SOFT_AES>(output + 2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

	xout0 = _mm_load_si128(output + 4);
	xout1 = _mm_load_si128(output + 5);
	xout2 = _mm_load_si128(output + 6);
	xout3 = _mm_load_si128(output + 7);
	xout4 = _mm_load_si128(output + 8);
	xout5 = _mm_load_si128(output + 9);
	xout6 = _mm_load_si128(output + 10);
	xout7 = _mm_load_si128(output + 11);

	for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8)
	{
		if(PREFETCH)
			_mm_prefetch((const char*)input + i + 0, _MM_HINT_NTA);

		xout0 = _mm_xor_si128(_mm_load_si128(input + i + 0), xout0);
		xout1 = _mm_xor_si128(_mm_load_si128(input + i + 1), xout1);
		xout2 = _mm_xor_si128(_mm_load_si128(input + i + 2), xout2);
		xout3 = _mm_xor_si128(_mm_load_si128(input + i + 3), xout3);

		if(PREFETCH)
			_mm_prefetch((const char*)input + i + 4, _MM_HINT_NTA);

		xout4 = _mm_xor_si128(_mm_load_si128(input + i + 4), xout4);
		xout5 = _mm_xor_si128(_mm_load_si128(input + i + 5), xout5);
		xout6 = _mm_xor_si128(_mm_load_si128(input + i + 6), xout6);
		xout7 = _mm_xor_si128(_mm_load_si128(input + i + 7), xout7);

		if(SOFT_AES)
		{
			soft_aes_round(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			soft_aes_round(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
		}
		else
		{
			aes_round(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			aes_round(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
		}

		if(ALGO == cryptonight_heavy || ALGO == cryptonight_haven || ALGO == cryptonight_bittube2)
			mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
	}

	if(ALGO == cryptonight_heavy || ALGO == cryptonight_haven || ALGO == cryptonight_bittube2)
	{
		for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8)
		{
			if(PREFETCH)
				_mm_prefetch((const char*)input + i + 0, _MM_HINT_NTA);

			xout0 = _mm_xor_si128(_mm_load_si128(input + i + 0), xout0);
			xout1 = _mm_xor_si128(_mm_load_si128(input + i + 1), xout1);
			xout2 = _mm_xor_si128(_mm_load_si128(input + i + 2), xout2);
			xout3 = _mm_xor_si128(_mm_load_si128(input + i + 3), xout3);

			if(PREFETCH)
				_mm_prefetch((const char*)input + i + 4, _MM_HINT_NTA);

			xout4 = _mm_xor_si128(_mm_load_si128(input + i + 4), xout4);
			xout5 = _mm_xor_si128(_mm_load_si128(input + i + 5), xout5);
			xout6 = _mm_xor_si128(_mm_load_si128(input + i + 6), xout6);
			xout7 = _mm_xor_si128(_mm_load_si128(input + i + 7), xout7);

			if(SOFT_AES)
			{
				soft_aes_round(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			}
			else
			{
				aes_round(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			}

			if(ALGO == cryptonight_heavy || ALGO == cryptonight_haven || ALGO == cryptonight_bittube2)
				mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
		}

		for(size_t i=0; i < 16; i++)
		{
			if(SOFT_AES)
			{
				soft_aes_round(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				soft_aes_round(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			}
			else
			{
				aes_round(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
				aes_round(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
			}

			mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
		}
	}

	_mm_store_si128(output + 4, xout0);
	_mm_store_si128(output + 5, xout1);
	_mm_store_si128(output + 6, xout2);
	_mm_store_si128(output + 7, xout3);
	_mm_store_si128(output + 8, xout4);
	_mm_store_si128(output + 9, xout5);
	_mm_store_si128(output + 10, xout6);
	_mm_store_si128(output + 11, xout7);
}

inline uint64_t int_sqrt33_1_double_precision(const uint64_t n0)
{
	__m128d x = _mm_castsi128_pd(_mm_add_epi64(_mm_cvtsi64_si128(n0 >> 12), _mm_set_epi64x(0, 1023ULL << 52)));
	x = _mm_sqrt_sd(_mm_setzero_pd(), x);
	uint64_t r = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_castpd_si128(x)));

	const uint64_t s = r >> 20;
	r >>= 19;

	uint64_t x2 = (s - (1022ULL << 32)) * (r - s - (1022ULL << 32) + 1);

#ifdef __INTEL_COMPILER
	_addcarry_u64(_subborrow_u64(0, x2, n0, (unsigned __int64*)&x2), r, 0, (unsigned __int64*)&r);
#elif defined(_MSC_VER) || (__GNUC__ >= 7)
	_addcarry_u64(_subborrow_u64(0, x2, n0, (unsigned long long int*)&x2), r, 0, (unsigned long long int*)&r);
#else
	// GCC versions prior to 7 don't generate correct assembly for _subborrow_u64 -> _addcarry_u64 sequence
	// Fallback to simpler code
	if (x2 < n0) ++r;
#endif
	return r;
}

inline __m128i aes_round_bittube2(const __m128i& val, const __m128i& key)
{
	alignas(16) uint32_t k[4];
	alignas(16) uint32_t x[4];
	_mm_store_si128((__m128i*)k, key);
	_mm_store_si128((__m128i*)x, _mm_xor_si128(val, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()))); // x = ~val
	#define BYTE(p, i) ((unsigned char*)&p)[i]
	k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^ saes_table[3][BYTE(x[3], 3)];
	x[0] ^= k[0];
	k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^ saes_table[3][BYTE(x[0], 3)];
	x[1] ^= k[1];
	k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^ saes_table[3][BYTE(x[1], 3)];
	x[2] ^= k[2];
	k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^ saes_table[3][BYTE(x[2], 3)];
	#undef BYTE
	return _mm_load_si128((__m128i*)k);
}

template<xmrstak_algo ALGO>
inline void cryptonight_monero_tweak(uint64_t* mem_out, __m128i tmp)
{
	mem_out[0] = _mm_cvtsi128_si64(tmp);

	tmp = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp), _mm_castsi128_ps(tmp)));
	uint64_t vh = _mm_cvtsi128_si64(tmp);

	uint8_t x = static_cast<uint8_t>(vh >> 24);
	static const uint16_t table = 0x7531;
	if(ALGO == cryptonight_monero || ALGO == cryptonight_aeon || ALGO == cryptonight_ipbc || ALGO == cryptonight_masari || ALGO == cryptonight_bittube2)
	{
		const uint8_t index = (((x >> 3) & 6) | (x & 1)) << 1;
		vh ^= ((table >> index) & 0x3) << 28;

		mem_out[1] = vh;
	}
	else if(ALGO == cryptonight_stellite)
	{
		const uint8_t index = (((x >> 4) & 6) | (x & 1)) << 1;
		vh ^= ((table >> index) & 0x3) << 28;

		mem_out[1] = vh;
	}

}

/** optimal type for sqrt
 *
 * Depending on the number of hashes calculated the optimal type for the sqrt value will be selected.
 *
 * @tparam N number of hashes per thread
 */
template<size_t N>
struct GetOptimalSqrtType
{
	using type = __m128i;
};

template<>
struct GetOptimalSqrtType<1u>
{
	using type = uint64_t;
};
template<size_t N>
using GetOptimalSqrtType_t = typename GetOptimalSqrtType<N>::type;

/** assign a value and convert if necessary
 *
 * @param output output type
 * @param input value which is assigned to output
 * @{
 */
inline void assign(__m128i& output, const uint64_t input)
{
	output = _mm_cvtsi64_si128(input);
}

inline void assign(uint64_t& output, const uint64_t input)
{
	output = input;
}

inline void assign(uint64_t& output, const __m128i& input)
{
	output = _mm_cvtsi128_si64(input);
}
/** @} */

inline void set_float_rounding_mode()
{
#ifdef _MSC_VER
	_control87(RC_DOWN, MCW_RC);
#else
	std::fesetround(FE_DOWNWARD);
#endif
}

#define CN_MONERO_V8_SHUFFLE_0(n, l0, idx0, ax0, bx0, bx1) \
	/* Shuffle the other 3x16 byte chunks in the current 64-byte cache line */ \
	if(ALGO == cryptonight_monero_v8) \
	{ \
		const uint64_t idx1 = idx0 & MASK; \
		const __m128i chunk1 = _mm_load_si128((__m128i *)&l0[idx1 ^ 0x10]); \
		const __m128i chunk2 = _mm_load_si128((__m128i *)&l0[idx1 ^ 0x20]); \
		const __m128i chunk3 = _mm_load_si128((__m128i *)&l0[idx1 ^ 0x30]); \
		_mm_store_si128((__m128i *)&l0[idx1 ^ 0x10], _mm_add_epi64(chunk3, bx1)); \
		_mm_store_si128((__m128i *)&l0[idx1 ^ 0x20], _mm_add_epi64(chunk1, bx0)); \
		_mm_store_si128((__m128i *)&l0[idx1 ^ 0x30], _mm_add_epi64(chunk2, ax0)); \
	}

#define CN_MONERO_V8_SHUFFLE_1(n, l0, idx0, ax0, bx0, bx1, lo, hi) \
	/* Shuffle the other 3x16 byte chunks in the current 64-byte cache line */ \
	if(ALGO == cryptonight_monero_v8) \
	{ \
		const uint64_t idx1 = idx0 & MASK; \
		const __m128i chunk1 = _mm_xor_si128(_mm_load_si128((__m128i *)&l0[idx1 ^ 0x10]), _mm_set_epi64x(lo, hi)); \
		const __m128i chunk2 = _mm_load_si128((__m128i *)&l0[idx1 ^ 0x20]); \
		hi ^= ((uint64_t*)&chunk2)[0]; \
		lo ^= ((uint64_t*)&chunk2)[1]; \
		const __m128i chunk3 = _mm_load_si128((__m128i *)&l0[idx1 ^ 0x30]); \
		_mm_store_si128((__m128i *)&l0[idx1 ^ 0x10], _mm_add_epi64(chunk3, bx1)); \
		_mm_store_si128((__m128i *)&l0[idx1 ^ 0x20], _mm_add_epi64(chunk1, bx0)); \
		_mm_store_si128((__m128i *)&l0[idx1 ^ 0x30], _mm_add_epi64(chunk2, ax0)); \
	}

#define CN_MONERO_V8_DIV(n, cx, sqrt_result, division_result_xmm, cl) \
	if(ALGO == cryptonight_monero_v8) \
	{ \
		uint64_t sqrt_result_tmp; \
		assign(sqrt_result_tmp, sqrt_result); \
		/* Use division and square root results from the _previous_ iteration to hide the latency */ \
		const uint64_t cx_64 = _mm_cvtsi128_si64(cx); \
		cl ^= static_cast<uint64_t>(_mm_cvtsi128_si64(division_result_xmm)) ^ (sqrt_result_tmp << 32); \
		const uint32_t d = (cx_64 + (sqrt_result_tmp << 1)) | 0x80000001UL; \
		/* Most and least significant bits in the divisor are set to 1 \
		 * to make sure we don't divide by a small or even number, \
		 * so there are no shortcuts for such cases \
		 * \
		 * Quotient may be as large as (2^64 - 1)/(2^31 + 1) = 8589934588 = 2^33 - 4 \
		 * We drop the highest bit to fit both quotient and remainder in 32 bits \
		 */  \
		/* Compiler will optimize it to a single div instruction */ \
		const uint64_t cx_s = _mm_cvtsi128_si64(_mm_srli_si128(cx, 8)); \
		const uint64_t division_result = static_cast<uint32_t>(cx_s / d) + ((cx_s % d) << 32); \
		division_result_xmm = _mm_cvtsi64_si128(static_cast<int64_t>(division_result)); \
		/* Use division_result as an input for the square root to prevent parallel implementation in hardware */ \
		assign(sqrt_result, int_sqrt33_1_double_precision(cx_64 + division_result)); \
	}

#define CN_INIT_SINGLE \
	if((ALGO == cryptonight_monero || ALGO == cryptonight_aeon || ALGO == cryptonight_ipbc || ALGO == cryptonight_stellite || ALGO == cryptonight_masari || ALGO == cryptonight_bittube2) && len < 43) \
	{ \
		memset(output, 0, 32 * N); \
		return; \
	}

#define CN_INIT(n, monero_const, l0, ax0, bx0, idx0, ptr0, bx1, sqrt_result, division_result_xmm) \
	keccak((const uint8_t *)input + len * n, len, ctx[n]->hash_state, 200); \
	uint64_t monero_const; \
	if(ALGO == cryptonight_monero || ALGO == cryptonight_aeon || ALGO == cryptonight_ipbc || ALGO == cryptonight_stellite || ALGO == cryptonight_masari || ALGO == cryptonight_bittube2) \
	{ \
		monero_const =  *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + len * n + 35); \
		monero_const ^=  *(reinterpret_cast<const uint64_t*>(ctx[n]->hash_state) + 24); \
	} \
	/* Optim - 99% time boundary */ \
	cn_explode_scratchpad<MEM, SOFT_AES, PREFETCH, ALGO>((__m128i*)ctx[n]->hash_state, (__m128i*)ctx[n]->long_state); \
	\
	__m128i ax0; \
	uint64_t idx0; \
	__m128i bx0; \
	uint8_t* l0 = ctx[n]->long_state; \
	/* BEGIN cryptonight_monero_v8 variables */ \
	__m128i bx1; \
	__m128i division_result_xmm; \
	GetOptimalSqrtType_t<N> sqrt_result; \
	/* END cryptonight_monero_v8 variables */ \
	{ \
		uint64_t* h0 = (uint64_t*)ctx[n]->hash_state; \
		idx0 = h0[0] ^ h0[4]; \
		ax0 = _mm_set_epi64x(h0[1] ^ h0[5], idx0); \
		bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]); \
		if(ALGO == cryptonight_monero_v8) \
		{ \
			bx1 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]); \
			division_result_xmm = _mm_cvtsi64_si128(h0[12]); \
			assign(sqrt_result, h0[13]); \
			set_float_rounding_mode(); \
		} \
	} \
	__m128i *ptr0

#define CN_STEP1(n, monero_const, l0, ax0, bx0, idx0, ptr0, cx, bx1) \
	__m128i cx; \
	ptr0 = (__m128i *)&l0[idx0 & MASK]; \
	cx = _mm_load_si128(ptr0); \
	if (ALGO == cryptonight_bittube2) \
	{ \
		cx = aes_round_bittube2(cx, ax0); \
	} \
	else \
	{ \
		if(SOFT_AES) \
			cx = soft_aesenc(cx, ax0); \
		else \
			cx = _mm_aesenc_si128(cx, ax0); \
	} \
	CN_MONERO_V8_SHUFFLE_0(n, l0, idx0, ax0, bx0, bx1)

#define CN_STEP2(n, monero_const, l0, ax0, bx0, idx0, ptr0, cx) \
	if(ALGO == cryptonight_monero || ALGO == cryptonight_aeon || ALGO == cryptonight_ipbc || ALGO == cryptonight_stellite || ALGO == cryptonight_masari || ALGO == cryptonight_bittube2) \
		cryptonight_monero_tweak<ALGO>((uint64_t*)ptr0, _mm_xor_si128(bx0, cx)); \
	else \
		_mm_store_si128((__m128i *)ptr0, _mm_xor_si128(bx0, cx)); \
	idx0 = _mm_cvtsi128_si64(cx); \
	\
	ptr0 = (__m128i *)&l0[idx0 & MASK]; \
	if(PREFETCH) \
		_mm_prefetch((const char*)ptr0, _MM_HINT_T0); \
	if(ALGO != cryptonight_monero_v8) \
		bx0 = cx

#define CN_STEP3(n, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0, cx, bx1, sqrt_result, division_result_xmm) \
	uint64_t lo, cl, ch; \
	uint64_t al0 = _mm_cvtsi128_si64(ax0); \
	uint64_t ah0 = ((uint64_t*)&ax0)[1]; \
	cl = ((uint64_t*)ptr0)[0]; \
	ch = ((uint64_t*)ptr0)[1]; \
	CN_MONERO_V8_DIV(n, cx, sqrt_result, division_result_xmm, cl); \
	{ \
		uint64_t hi; \
		lo = _umul128(idx0, cl, &hi); \
		CN_MONERO_V8_SHUFFLE_1(n, l0, idx0, ax0, bx0, bx1, lo, hi); \
		ah0 += lo; \
		al0 += hi; \
	} \
	if(ALGO == cryptonight_monero_v8) \
	{ \
		bx1 = bx0; \
		bx0 = cx; \
	} \
	((uint64_t*)ptr0)[0] = al0; \
	if(PREFETCH) \
		_mm_prefetch((const char*)ptr0, _MM_HINT_T0)

#define CN_STEP4(n, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0) \
	if (ALGO == cryptonight_monero || ALGO == cryptonight_aeon || ALGO == cryptonight_ipbc || ALGO == cryptonight_stellite || ALGO == cryptonight_masari || ALGO == cryptonight_bittube2) \
	{ \
		if (ALGO == cryptonight_ipbc || ALGO == cryptonight_bittube2) \
			((uint64_t*)ptr0)[1] = ah0 ^ monero_const ^ ((uint64_t*)ptr0)[0]; \
		else \
			((uint64_t*)ptr0)[1] = ah0 ^ monero_const; \
	} \
	else \
		((uint64_t*)ptr0)[1] = ah0; \
	al0 ^= cl; \
	ah0 ^= ch; \
	ax0 = _mm_set_epi64x(ah0, al0); \
	idx0 = al0;

#define CN_STEP5(n, monero_const, l0, ax0, bx0, idx0, ptr0) \
	if(ALGO == cryptonight_heavy || ALGO == cryptonight_bittube2) \
	{ \
		ptr0 = (__m128i *)&l0[idx0 & MASK]; \
		int64_t u  = ((int64_t*)ptr0)[0]; \
		int32_t d  = ((int32_t*)ptr0)[2]; \
		int64_t q = u / (d | 0x5); \
		\
		((int64_t*)ptr0)[0] = u ^ q; \
		idx0 = d ^ q; \
	} \
	else if(ALGO == cryptonight_haven) \
	{ \
		ptr0 = (__m128i *)&l0[idx0 & MASK]; \
		int64_t u  = ((int64_t*)ptr0)[0]; \
		int32_t d  = ((int32_t*)ptr0)[2]; \
		int64_t q = u / (d | 0x5); \
		\
		((int64_t*)ptr0)[0] = u ^ q; \
		idx0 = (~d) ^ q; \
	}

#define CN_FINALIZE(n) \
	/* Optim - 90% time boundary */ \
	cn_implode_scratchpad<MEM, SOFT_AES, PREFETCH, ALGO>((__m128i*)ctx[n]->long_state, (__m128i*)ctx[n]->hash_state); \
	/* Optim - 99% time boundary */ \
	keccakf((uint64_t*)ctx[n]->hash_state, 24); \
	extra_hashes[ctx[n]->hash_state[0] & 3](ctx[n]->hash_state, 200, (char*)output + 32 * n)

//! defer the evaluation of an macro
#ifndef _MSC_VER
#	define CN_DEFER(...) __VA_ARGS__
#else
#	define CN_EMPTY(...)
#	define CN_DEFER(...) __VA_ARGS__ CN_EMPTY()
#endif

//! execute the macro f with the passed arguments
#define CN_EXEC(f,...) CN_DEFER(f)(__VA_ARGS__)

/** add append n to all arguments and keeps n as first argument
 *
 * @param n number which is appended to the arguments (expect the first argument n)
 *
 * @code{.cpp}
 * CN_ENUM_2(1, foo, bar)
 * // is transformed to
 * 1, foo1, bar1
 * @endcode
 */
#define CN_ENUM_0(n, ...) n
#define CN_ENUM_1(n, x1) n, x1 ## n
#define CN_ENUM_2(n, x1, x2) n, x1 ## n, x2 ## n
#define CN_ENUM_3(n, x1, x2, x3) n, x1 ## n, x2 ## n, x3 ## n
#define CN_ENUM_4(n, x1, x2, x3, x4) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n
#define CN_ENUM_5(n, x1, x2, x3, x4, x5) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n
#define CN_ENUM_6(n, x1, x2, x3, x4, x5, x6) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n
#define CN_ENUM_7(n, x1, x2, x3, x4, x5, x6, x7) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n
#define CN_ENUM_8(n, x1, x2, x3, x4, x5, x6, x7, x8) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n
#define CN_ENUM_9(n, x1, x2, x3, x4, x5, x6, x7, x8, x9) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n
#define CN_ENUM_10(n, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n, x10 ## n
#define CN_ENUM_11(n, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n, x10 ## n, x11 ## n
#define CN_ENUM_12(n, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n, x10 ## n, x11 ## n, x12 ## n
#define CN_ENUM_13(n, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n, x10 ## n, x11 ## n, x12 ## n, x13 ## n
#define CN_ENUM_14(n, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n, x10 ## n, x11 ## n, x12 ## n, x13 ## n, x14 ## n
#define CN_ENUM_15(n, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) n, x1 ## n, x2 ## n, x3 ## n, x4 ## n, x5 ## n, x6 ## n, x7 ## n, x8 ## n, x9 ## n, x10 ## n, x11 ## n, x12 ## n, x13 ## n, x14 ## n, x15 ## n

/** repeat a macro call multiple times
 *
 * @param n number of arguments followed after f
 * @param f name of the macro which should be executed
 * @param ... n parameter which name will get appended by a unique number
 *
 * @code{.cpp}
 * REPEAT_2(2, f, foo, bar)
 * // is transformed to
 * f(0, foo0, bar); f(1, foo1, bar1)
 * @endcode
 */
#define REPEAT_1(n, f, ...) CN_EXEC(f, CN_ENUM_ ## n(0, __VA_ARGS__))
#define REPEAT_2(n, f, ...) CN_EXEC(f, CN_ENUM_ ## n(0, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(1, __VA_ARGS__))
#define REPEAT_3(n, f, ...) CN_EXEC(f, CN_ENUM_ ## n(0, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(1, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(2, __VA_ARGS__))
#define REPEAT_4(n, f, ...) CN_EXEC(f, CN_ENUM_ ## n(0, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(1, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(2, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(3, __VA_ARGS__))
#define REPEAT_5(n, f, ...) CN_EXEC(f, CN_ENUM_ ## n(0, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(1, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(2, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(3, __VA_ARGS__)); CN_EXEC(f, CN_ENUM_ ## n(4, __VA_ARGS__))

template< size_t N>
struct Cryptonight_hash;

template< >
struct Cryptonight_hash<1>
{
	static constexpr size_t N = 1;

	template<xmrstak_algo ALGO, bool SOFT_AES, bool PREFETCH>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MASK = cn_select_mask<ALGO>();
		constexpr size_t ITERATIONS = cn_select_iter<ALGO>();
		constexpr size_t MEM = cn_select_memory<ALGO>();

		CN_INIT_SINGLE;
		REPEAT_1(9, CN_INIT, monero_const, l0, ax0, bx0, idx0, ptr0, bx1, sqrt_result, division_result_xmm);

		// Optim - 90% time boundary
		for(size_t i = 0; i < ITERATIONS; i++)
		{
			REPEAT_1(8, CN_STEP1, monero_const, l0, ax0, bx0, idx0, ptr0, cx, bx1);
			REPEAT_1(7, CN_STEP2, monero_const, l0, ax0, bx0, idx0, ptr0, cx);
			REPEAT_1(15, CN_STEP3, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0, cx, bx1, sqrt_result, division_result_xmm);
			REPEAT_1(11, CN_STEP4, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0);
			REPEAT_1(6, CN_STEP5, monero_const, l0, ax0, bx0, idx0, ptr0);
		}

		REPEAT_1(0, CN_FINALIZE);
	}
};

template< >
struct Cryptonight_hash<2>
{
	static constexpr size_t N = 2;

	template<xmrstak_algo ALGO, bool SOFT_AES, bool PREFETCH>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MASK = cn_select_mask<ALGO>();
		constexpr size_t ITERATIONS = cn_select_iter<ALGO>();
		constexpr size_t MEM = cn_select_memory<ALGO>();

		CN_INIT_SINGLE;
		REPEAT_2(9, CN_INIT, monero_const, l0, ax0, bx0, idx0, ptr0, bx1, sqrt_result, division_result_xmm);

		// Optim - 90% time boundary
		for(size_t i = 0; i < ITERATIONS; i++)
		{
			REPEAT_2(8, CN_STEP1, monero_const, l0, ax0, bx0, idx0, ptr0, cx, bx1);
			REPEAT_2(7, CN_STEP2, monero_const, l0, ax0, bx0, idx0, ptr0, cx);
			REPEAT_2(15, CN_STEP3, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0, cx, bx1, sqrt_result, division_result_xmm);
			REPEAT_2(11, CN_STEP4, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0);
			REPEAT_2(6, CN_STEP5, monero_const, l0, ax0, bx0, idx0, ptr0);
		}

		REPEAT_2(0, CN_FINALIZE);
	}
};

template< >
struct Cryptonight_hash<3>
{
	static constexpr size_t N = 3;

	template<xmrstak_algo ALGO, bool SOFT_AES, bool PREFETCH>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MASK = cn_select_mask<ALGO>();
		constexpr size_t ITERATIONS = cn_select_iter<ALGO>();
		constexpr size_t MEM = cn_select_memory<ALGO>();

		CN_INIT_SINGLE;
		REPEAT_3(9, CN_INIT, monero_const, l0, ax0, bx0, idx0, ptr0, bx1, sqrt_result, division_result_xmm);

		// Optim - 90% time boundary
		for(size_t i = 0; i < ITERATIONS; i++)
		{
			REPEAT_3(8, CN_STEP1, monero_const, l0, ax0, bx0, idx0, ptr0, cx, bx1);
			REPEAT_3(7, CN_STEP2, monero_const, l0, ax0, bx0, idx0, ptr0, cx);
			REPEAT_3(15, CN_STEP3, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0, cx, bx1, sqrt_result, division_result_xmm);
			REPEAT_3(11, CN_STEP4, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0);
			REPEAT_3(6, CN_STEP5, monero_const, l0, ax0, bx0, idx0, ptr0);
		}

		REPEAT_3(0, CN_FINALIZE);
	}
};

template< >
struct Cryptonight_hash<4>
{
	static constexpr size_t N = 4;

	template<xmrstak_algo ALGO, bool SOFT_AES, bool PREFETCH>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MASK = cn_select_mask<ALGO>();
		constexpr size_t ITERATIONS = cn_select_iter<ALGO>();
		constexpr size_t MEM = cn_select_memory<ALGO>();

		CN_INIT_SINGLE;
		REPEAT_4(9, CN_INIT, monero_const, l0, ax0, bx0, idx0, ptr0, bx1, sqrt_result, division_result_xmm);

		// Optim - 90% time boundary
		for(size_t i = 0; i < ITERATIONS; i++)
		{
			REPEAT_4(8, CN_STEP1, monero_const, l0, ax0, bx0, idx0, ptr0, cx, bx1);
			REPEAT_4(7, CN_STEP2, monero_const, l0, ax0, bx0, idx0, ptr0, cx);
			REPEAT_4(15, CN_STEP3, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0, cx, bx1, sqrt_result, division_result_xmm);
			REPEAT_4(11, CN_STEP4, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0);
			REPEAT_4(6, CN_STEP5, monero_const, l0, ax0, bx0, idx0, ptr0);
		}

		REPEAT_4(0, CN_FINALIZE);
	}
};

template< >
struct Cryptonight_hash<5>
{
	static constexpr size_t N = 5;

	template<xmrstak_algo ALGO, bool SOFT_AES, bool PREFETCH>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MASK = cn_select_mask<ALGO>();
		constexpr size_t ITERATIONS = cn_select_iter<ALGO>();
		constexpr size_t MEM = cn_select_memory<ALGO>();

		CN_INIT_SINGLE;
		REPEAT_5(9, CN_INIT, monero_const, l0, ax0, bx0, idx0, ptr0, bx1, sqrt_result, division_result_xmm);

		// Optim - 90% time boundary
		for(size_t i = 0; i < ITERATIONS; i++)
		{
			REPEAT_5(8, CN_STEP1, monero_const, l0, ax0, bx0, idx0, ptr0, cx, bx1);
			REPEAT_5(7, CN_STEP2, monero_const, l0, ax0, bx0, idx0, ptr0, cx);
			REPEAT_5(15, CN_STEP3, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0, cx, bx1, sqrt_result, division_result_xmm);
			REPEAT_5(11, CN_STEP4, monero_const, l0, ax0, bx0, idx0, ptr0, lo, cl, ch, al0, ah0);
			REPEAT_5(6, CN_STEP5, monero_const, l0, ax0, bx0, idx0, ptr0);
		}

		REPEAT_5(0, CN_FINALIZE);
	}
};

extern "C" void cryptonight_v8_mainloop_ivybridge_asm(cryptonight_ctx* ctx0);
extern "C" void cryptonight_v8_mainloop_ryzen_asm(cryptonight_ctx* ctx0);
extern "C" void cryptonight_v8_double_mainloop_sandybridge_asm(cryptonight_ctx* ctx0, cryptonight_ctx* ctx1);


template< size_t N, size_t asm_version>
struct Cryptonight_hash_asm;

template<size_t asm_version>
struct Cryptonight_hash_asm<1, asm_version>
{
	static constexpr size_t N = 1;

	template<xmrstak_algo ALGO>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MEM = cn_select_memory<ALGO>();

		keccak((const uint8_t *)input, len, ctx[0]->hash_state, 200);
		cn_explode_scratchpad<MEM, false, false, ALGO>((__m128i*)ctx[0]->hash_state, (__m128i*)ctx[0]->long_state);

		if(asm_version == 0)
			cryptonight_v8_mainloop_ivybridge_asm(ctx[0]);
		else if(asm_version == 1)
			cryptonight_v8_mainloop_ryzen_asm(ctx[0]);

		cn_implode_scratchpad<MEM, false, false, ALGO>((__m128i*)ctx[0]->long_state, (__m128i*)ctx[0]->hash_state);
		keccakf((uint64_t*)ctx[0]->hash_state, 24);
		extra_hashes[ctx[0]->hash_state[0] & 3](ctx[0]->hash_state, 200, (char*)output);
	}
};

// double hash only for intel
template< >
struct Cryptonight_hash_asm<2, 0>
{
	static constexpr size_t N = 2;

	template<xmrstak_algo ALGO>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx)
	{
		constexpr size_t MEM = cn_select_memory<ALGO>();

		for(size_t i = 0; i < N; ++i)
		{
			keccak((const uint8_t *)input + len * i, len, ctx[i]->hash_state, 200);
			/* Optim - 99% time boundary */
			cn_explode_scratchpad<MEM, false, false, ALGO>((__m128i*)ctx[i]->hash_state, (__m128i*)ctx[i]->long_state);
		}

		cryptonight_v8_double_mainloop_sandybridge_asm(ctx[0], ctx[1]);

		for(size_t i = 0; i < N; ++i)
		{
			/* Optim - 90% time boundary */
			cn_implode_scratchpad<MEM, false, false, ALGO>((__m128i*)ctx[i]->long_state, (__m128i*)ctx[i]->hash_state);
			/* Optim - 99% time boundary */
			keccakf((uint64_t*)ctx[i]->hash_state, 24);
			extra_hashes[ctx[i]->hash_state[0] & 3](ctx[i]->hash_state, 200, (char*)output + 32 * i);
		}
	}
};
