#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_extra.hpp"
#include "cuda_keccak.hpp"

namespace xmrstak
{
namespace nvidia
{

struct __m128i : public int4
{

	__forceinline__ __device__ __m128i() {}

	__forceinline__ __device__ __m128i(
		const uint32_t x0, const uint32_t x1,
		const uint32_t x2, const uint32_t x3)
	{
		x = x0;
		y = x1;
		z = x2;
		w = x3;
	}

	__forceinline__ __device__ __m128i(const int x0)
	{
		x = x0;
		y = x0;
		z = x0;
		w = x0;
	}

	__forceinline__ __device__ __m128i operator|(const __m128i& other)
	{
		return __m128i(
			x | other.x,
			y | other.y,
			z | other.z,
			w | other.w);
	}

	__forceinline__ __device__ __m128i operator^(const __m128i& other)
	{
		return __m128i(
			x ^ other.x,
			y ^ other.y,
			z ^ other.z,
			w ^ other.w);
	}
};

struct __m128 : public float4
{

	__forceinline__ __device__ __m128() {}

	__forceinline__ __device__ __m128(
		const float x0, const float x1,
		const float x2, const float x3)
	{
		float4::x = x0;
		float4::y = x1;
		float4::z = x2;
		float4::w = x3;
	}

	__forceinline__ __device__ __m128(const float x0)
	{
		float4::x = x0;
		float4::y = x0;
		float4::z = x0;
		float4::w = x0;
	}

	__forceinline__ __device__ __m128(const __m128i& x0)
	{
		float4::x = int2float(x0.x);
		float4::y = int2float(x0.y);
		float4::z = int2float(x0.z);
		float4::w = int2float(x0.w);
	}

	__forceinline__ __device__ __m128i get_int()
	{
		return __m128i(
			(int)x,
			(int)y,
			(int)z,
			(int)w);
	}

	__forceinline__ __device__ __m128 operator+(const __m128& other)
	{
		return __m128(
			x + other.x,
			y + other.y,
			z + other.z,
			w + other.w);
	}

	__forceinline__ __device__ __m128 operator-(const __m128& other)
	{
		return __m128(
			x - other.x,
			y - other.y,
			z - other.z,
			w - other.w);
	}

	__forceinline__ __device__ __m128 operator*(const __m128& other)
	{
		return __m128(
			x * other.x,
			y * other.y,
			z * other.z,
			w * other.w);
	}

	__forceinline__ __device__ __m128 operator/(const __m128& other)
	{
		return __m128(
			x / other.x,
			y / other.y,
			z / other.z,
			w / other.w);
	}

	__forceinline__ __device__ __m128& trunc()
	{
		x = ::truncf(x);
		y = ::truncf(y);
		z = ::truncf(z);
		w = ::truncf(w);

		return *this;
	}

	__forceinline__ __device__ __m128& abs()
	{
		x = ::fabsf(x);
		y = ::fabsf(y);
		z = ::fabsf(z);
		w = ::fabsf(w);

		return *this;
	}

	__forceinline__ __device__ __m128& floor()
	{
		x = ::floorf(x);
		y = ::floorf(y);
		z = ::floorf(z);
		w = ::floorf(w);

		return *this;
	}
};

template <typename T>
__device__ void print(const char* name, T value)
{
	printf("g %s: ", name);
	for(int i = 0; i < 4; ++i)
	{
		printf("%08X ", ((uint32_t*)&value)[i]);
	}
	printf("\n");
}

template <>
__device__ void print<__m128>(const char* name, __m128 value)
{
	printf("g %s: ", name);
	for(int i = 0; i < 4; ++i)
	{
		printf("%f ", ((float*)&value)[i]);
	}
	printf("\n");
}

#define SHOW(name) print(#name, name)

__forceinline__ __device__ __m128 _mm_add_ps(__m128 a, __m128 b)
{
	return a + b;
}

__forceinline__ __device__ __m128 _mm_sub_ps(__m128 a, __m128 b)
{
	return a - b;
}

__forceinline__ __device__ __m128 _mm_mul_ps(__m128 a, __m128 b)
{
	return a * b;
}

__forceinline__ __device__ __m128 _mm_div_ps(__m128 a, __m128 b)
{
	return a / b;
}

__forceinline__ __device__ __m128 _mm_and_ps(__m128 a, int b)
{
	return __m128(
		int_as_float(float_as_int(a.x) & b),
		int_as_float(float_as_int(a.y) & b),
		int_as_float(float_as_int(a.z) & b),
		int_as_float(float_as_int(a.w) & b));
}

__forceinline__ __device__ __m128 _mm_or_ps(__m128 a, int b)
{
	return __m128(
		int_as_float(float_as_int(a.x) | b),
		int_as_float(float_as_int(a.y) | b),
		int_as_float(float_as_int(a.z) | b),
		int_as_float(float_as_int(a.w) | b));
}

__forceinline__ __device__ __m128 _mm_xor_ps(__m128 a, int b)
{
	return __m128(
		int_as_float(float_as_int(a.x) ^ b),
		int_as_float(float_as_int(a.y) ^ b),
		int_as_float(float_as_int(a.z) ^ b),
		int_as_float(float_as_int(a.w) ^ b));
}

__forceinline__ __device__ __m128 _mm_fmod_ps(__m128 v, float dc)
{
	__m128 d(dc);
	__m128 c = _mm_div_ps(v, d);
	c.trunc(); //_mm_round_ps(c, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
	// c = _mm_cvtepi32_ps(_mm_cvttps_epi32(c)); - sse2
	c = _mm_mul_ps(c, d);
	return _mm_sub_ps(v, c);

	//return a.fmodf(b);
}

__forceinline__ __device__ __m128i _mm_xor_si128(__m128i a, __m128i b)
{
	return a ^ b;
}

__forceinline__ __device__ __m128i _mm_alignr_epi8(__m128i a, const uint32_t rot)
{
	const uint32_t right = 8 * rot;
	const uint32_t left = (32 - 8 * rot);
	return __m128i(
		((uint32_t)a.x >> right) | (a.y << left),
		((uint32_t)a.y >> right) | (a.z << left),
		((uint32_t)a.z >> right) | (a.w << left),
		((uint32_t)a.w >> right) | (a.x << left));
}

__device__ __m128i* scratchpad_ptr(uint32_t idx, uint32_t n, int* lpad, const uint32_t MASK) { return (__m128i*)((uint8_t*)lpad + (idx & MASK) + n * 16); }

__forceinline__ __device__ __m128 fma_break(__m128 x)
{
	// Break the dependency chain by setitng the exp to ?????01
	x = _mm_and_ps(x, 0xFEFFFFFF);
	return _mm_or_ps(x, 0x00800000);
}

// 9
__forceinline__ __device__ void sub_round(__m128 n0, __m128 n1, __m128 n2, __m128 n3, __m128 rnd_c, __m128& n, __m128& d, __m128& c)
{
	n1 = _mm_add_ps(n1, c);
	__m128 nn = _mm_mul_ps(n0, c);
	nn = _mm_mul_ps(n1, _mm_mul_ps(nn, nn));
	nn = fma_break(nn);
	n = _mm_add_ps(n, nn);

	n3 = _mm_sub_ps(n3, c);
	__m128 dd = _mm_mul_ps(n2, c);
	dd = _mm_mul_ps(n3, _mm_mul_ps(dd, dd));
	dd = fma_break(dd);
	d = _mm_add_ps(d, dd);

	//Constant feedback
	c = _mm_add_ps(c, rnd_c);
	c = _mm_add_ps(c, 0.734375f);
	__m128 r = _mm_add_ps(nn, dd);
	r = _mm_and_ps(r, 0x807FFFFF);
	r = _mm_or_ps(r, 0x40000000);
	c = _mm_add_ps(c, r);
}

// 9*8 + 2 = 74
__forceinline__ __device__ void round_compute(__m128 n0, __m128 n1, __m128 n2, __m128 n3, __m128 rnd_c, __m128& c, __m128& r)
{
	__m128 n(0.0f), d(0.0f);

	sub_round(n0, n1, n2, n3, rnd_c, n, d, c);
	sub_round(n1, n2, n3, n0, rnd_c, n, d, c);
	sub_round(n2, n3, n0, n1, rnd_c, n, d, c);
	sub_round(n3, n0, n1, n2, rnd_c, n, d, c);
	sub_round(n3, n2, n1, n0, rnd_c, n, d, c);
	sub_round(n2, n1, n0, n3, rnd_c, n, d, c);
	sub_round(n1, n0, n3, n2, rnd_c, n, d, c);
	sub_round(n0, n3, n2, n1, rnd_c, n, d, c);

	// Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
	d = _mm_and_ps(d, 0xFF7FFFFF);
	d = _mm_or_ps(d, 0x40000000);
	r = _mm_add_ps(r, _mm_div_ps(n, d));
}

// 74*8 = 595
__forceinline__ __device__ __m128i single_comupte(__m128 n0, __m128 n1, __m128 n2, __m128 n3, float cnt, __m128 rnd_c, __m128& sum)
{
	__m128 c(cnt);
	// 35 maths calls follow (140 FLOPS)
	__m128 r = __m128(0.0f);
	for(int i = 0; i < 4; ++i)
		round_compute(n0, n1, n2, n3, rnd_c, c, r);
	// do a quick fmod by setting exp to 2
	r = _mm_and_ps(r, 0x807FFFFF);
	r = _mm_or_ps(r, 0x40000000);
	sum = r;								 // 34
	r = _mm_mul_ps(r, __m128(536870880.0f)); // 35
	return r.get_int();
}

__forceinline__ __device__ void single_comupte_wrap(const uint32_t rot, const __m128i& v0, const __m128i& v1, const __m128i& v2, const __m128i& v3, float cnt, __m128 rnd_c, __m128& sum, __m128i& out)
{
	__m128 n0(v0);
	__m128 n1(v1);
	__m128 n2(v2);
	__m128 n3(v3);

	__m128i r = single_comupte(n0, n1, n2, n3, cnt, rnd_c, sum);
	out = rot == 0 ? r : _mm_alignr_epi8(r, rot);
}

__constant__ uint32_t look[16][4] = {
	{0, 1, 2, 3},
	{0, 2, 3, 1},
	{0, 3, 1, 2},
	{0, 3, 2, 1},

	{1, 0, 2, 3},
	{1, 2, 3, 0},
	{1, 3, 0, 2},
	{1, 3, 2, 0},

	{2, 1, 0, 3},
	{2, 0, 3, 1},
	{2, 3, 1, 0},
	{2, 3, 0, 1},

	{3, 1, 2, 0},
	{3, 2, 0, 1},
	{3, 0, 1, 2},
	{3, 0, 2, 1}};

__constant__ float ccnt[16] = {
	1.34375f,
	1.28125f,
	1.359375f,
	1.3671875f,

	1.4296875f,
	1.3984375f,
	1.3828125f,
	1.3046875f,

	1.4140625f,
	1.2734375f,
	1.2578125f,
	1.2890625f,

	1.3203125f,
	1.3515625f,
	1.3359375f,
	1.4609375f};

__forceinline__ __device__ void sync()
{
#if(__CUDACC_VER_MAJOR__ >= 9)
	__syncwarp();
#else
	__syncthreads();
#endif
}

struct SharedMemChunk
{
	__m128i out[16];
	__m128 va[16];
};

__global__ void cryptonight_core_gpu_phase2_gpu(
	const uint32_t ITERATIONS, const size_t MEMORY, const uint32_t MASK,
	int32_t* spad, int* lpad_in, int bfactor, int partidx, uint32_t* roundVs, uint32_t* roundS)
{

	const int batchsize = (ITERATIONS * 2) >> (1 + bfactor);

	extern __shared__ SharedMemChunk smemExtern_in[];

	const uint32_t chunk = threadIdx.x / 16;
	const uint32_t numHashPerBlock = blockDim.x / 16;

	int* lpad = (int*)((uint8_t*)lpad_in + size_t(MEMORY) * (blockIdx.x * numHashPerBlock + chunk));

	SharedMemChunk* smem = smemExtern_in + chunk;

	uint32_t tid = threadIdx.x % 16;

	const uint32_t idxHash = blockIdx.x * numHashPerBlock + threadIdx.x / 16;
	uint32_t s = 0;

	__m128 vs(0);
	if(partidx != 0)
	{
		vs = ((__m128*)roundVs)[idxHash];
		s = roundS[idxHash];
	}
	else
	{
		s = ((uint32_t*)spad)[idxHash * 50] >> 8;
	}

	// tid divided
	const uint32_t tidd = tid / 4;
	// tid modulo
	const uint32_t tidm = tid % 4;
	const uint32_t block = tidd * 16 + tidm;

	for(size_t i = 0; i < batchsize; i++)
	{
		sync();
		int tmp = ((int*)scratchpad_ptr(s, tidd, lpad, MASK))[tidm];
		((int*)smem->out)[tid] = tmp;
		sync();

		__m128 rc = vs;
		single_comupte_wrap(
			tidm,
			*(smem->out + look[tid][0]),
			*(smem->out + look[tid][1]),
			*(smem->out + look[tid][2]),
			*(smem->out + look[tid][3]),
			ccnt[tid], rc, smem->va[tid],
			smem->out[tid]);

		sync();

		int outXor = ((int*)smem->out)[block];
		for(uint32_t dd = block + 4; dd < (tidd + 1) * 16; dd += 4)
			outXor ^= ((int*)smem->out)[dd];

		((int*)scratchpad_ptr(s, tidd, lpad, MASK))[tidm] = outXor ^ tmp;
		((int*)smem->out)[tid] = outXor;

		float va_tmp1 = ((float*)smem->va)[block] + ((float*)smem->va)[block + 4];
		float va_tmp2 = ((float*)smem->va)[block + 8] + ((float*)smem->va)[block + 12];
		((float*)smem->va)[tid] = va_tmp1 + va_tmp2;

		sync();

		__m128i out2 = smem->out[0] ^ smem->out[1] ^ smem->out[2] ^ smem->out[3];
		va_tmp1 = ((float*)smem->va)[block] + ((float*)smem->va)[block + 4];
		va_tmp2 = ((float*)smem->va)[block + 8] + ((float*)smem->va)[block + 12];
		((float*)smem->va)[tid] = va_tmp1 + va_tmp2;

		sync();

		vs = smem->va[0];
		vs.abs(); // take abs(va) by masking the float sign bit
		auto xx = _mm_mul_ps(vs, __m128(16777216.0f));
		// vs range 0 - 64
		auto xx_int = xx.get_int();
		out2 = _mm_xor_si128(xx_int, out2);
		// vs is now between 0 and 1
		vs = _mm_div_ps(vs, __m128(64.0f));
		s = out2.x ^ out2.y ^ out2.z ^ out2.w;
	}
	if(partidx != ((1 << bfactor) - 1) && threadIdx.x % 16 == 0)
	{
		const uint32_t numHashPerBlock2 = blockDim.x / 16;
		const uint32_t idxHash2 = blockIdx.x * numHashPerBlock2 + threadIdx.x / 16;
		((__m128*)roundVs)[idxHash2] = vs;
		roundS[idxHash2] = s;
	}
}

__forceinline__ __device__ void generate_512(uint64_t idx, const uint64_t* in, uint8_t* out)
{
	uint64_t hash[25];

	hash[0] = in[0] ^ idx;
#pragma unroll 24
	for(int i = 1; i < 25; ++i)
		hash[i] = in[i];

	cn_keccakf2(hash);
#pragma unroll 10
	for(int i = 0; i < 10; ++i)
		((ulonglong2*)out)[i] = ((ulonglong2*)hash)[i];
	out += 160;

	cn_keccakf2(hash);
#pragma unroll 11
	for(int i = 0; i < 11; ++i)
		((ulonglong2*)out)[i] = ((ulonglong2*)hash)[i];
	out += 176;

	cn_keccakf2(hash);
#pragma unroll 11
	for(int i = 0; i < 11; ++i)
		((ulonglong2*)out)[i] = ((ulonglong2*)hash)[i];
}

__global__ void cn_explode_gpu(const size_t MEMORY, int32_t* spad_in, int* lpad_in)
{
	__shared__ uint64_t state[25];

	uint8_t* lpad = (uint8_t*)lpad_in + blockIdx.x * MEMORY;
	uint64_t* spad = (uint64_t*)((uint8_t*)spad_in + blockIdx.x * 200);

	for(int i = threadIdx.x; i < 25; i += blockDim.x)
		state[i] = spad[i];

	sync();

	for(uint64_t i = threadIdx.x; i < MEMORY / 512; i += blockDim.x)
	{
		generate_512(i, state, (uint8_t*)lpad + i * 512);
	}
}

} // namespace nvidia
} // namespace xmrstak
