R"===(


inline float4 _mm_add_ps(float4 a, float4 b)
{
	return a + b;
}

inline float4 _mm_sub_ps(float4 a, float4 b)
{
	return a - b;
}

inline float4 _mm_mul_ps(float4 a, float4 b)
{

	//#pragma OPENCL SELECT_ROUNDING_MODE rte
	return a * b;
}

inline float4 _mm_div_ps(float4 a, float4 b)
{
	return a / b;
}

inline float4 _mm_and_ps(float4 a, int b)
{
	return as_float4(as_int4(a) & (int4)(b));
}

inline float4 _mm_or_ps(float4 a, int b)
{
	return as_float4(as_int4(a) | (int4)(b));
}

inline float4 _mm_fmod_ps(float4 v, float dc)
{
	float4 d = (float4)(dc);
	float4 c = _mm_div_ps(v, d);
	c = trunc(c);
	c = _mm_mul_ps(c, d);
	return _mm_sub_ps(v, c);
}

inline int4 _mm_xor_si128(int4 a, int4 b)
{
	return a ^ b;
}

inline float4 _mm_xor_ps(float4 a, int b)
{
	return as_float4(as_int4(a) ^ (int4)(b));
}

inline int4 _mm_alignr_epi8(int4 a, const uint rot)
{
	const uint right = 8 * rot;
	const uint left = (32 - 8 * rot);
	return (int4)(
		((uint)a.x >> right) | ( a.y << left ),
		((uint)a.y >> right) | ( a.z << left ),
		((uint)a.z >> right) | ( a.w << left ),
		((uint)a.w >> right) | ( a.x << left )
	);
}


inline global int4* scratchpad_ptr(uint idx, uint n, __global int *lpad) { return (__global int4*)((__global char*)lpad + (idx & MASK) + n * 16); }

inline float4 fma_break(float4 x)
{
	// Break the dependency chain by setitng the exp to ?????01
	x = _mm_and_ps(x, 0xFEFFFFFF);
	return _mm_or_ps(x, 0x00800000);
}

inline void sub_round(float4 n0, float4 n1, float4 n2, float4 n3, float4 rnd_c, float4* n, float4* d, float4* c)
{
	n1 = _mm_add_ps(n1, *c);
	float4 nn = _mm_mul_ps(n0, *c);
	nn = _mm_mul_ps(n1, _mm_mul_ps(nn,nn));
	nn = fma_break(nn);
	*n = _mm_add_ps(*n, nn);

	n3 = _mm_sub_ps(n3, *c);
	float4 dd = _mm_mul_ps(n2, *c);
	dd = _mm_mul_ps(n3, _mm_mul_ps(dd,dd));
	dd = fma_break(dd);
	*d = _mm_add_ps(*d, dd);

	//Constant feedback
	*c = _mm_add_ps(*c, rnd_c);
	*c = _mm_add_ps(*c, (float4)(0.734375f));
	float4 r = _mm_add_ps(nn, dd);
	r = _mm_and_ps(r, 0x807FFFFF);
	r = _mm_or_ps(r, 0x40000000);
	*c = _mm_add_ps(*c, r);

}

// 9*8 + 2 = 74
inline void round_compute(float4 n0, float4 n1, float4 n2, float4 n3, float4 rnd_c, float4* c, float4* r)
{
	float4 n = (float4)(0.0f);
	float4 d = (float4)(0.0f);

	sub_round(n0, n1, n2, n3, rnd_c, &n, &d, c);
	sub_round(n1, n2, n3, n0, rnd_c, &n, &d, c);
	sub_round(n2, n3, n0, n1, rnd_c, &n, &d, c);
	sub_round(n3, n0, n1, n2, rnd_c, &n, &d, c);
	sub_round(n3, n2, n1, n0, rnd_c, &n, &d, c);
	sub_round(n2, n1, n0, n3, rnd_c, &n, &d, c);
	sub_round(n1, n0, n3, n2, rnd_c, &n, &d, c);
	sub_round(n0, n3, n2, n1, rnd_c, &n, &d, c);

	// Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
	d = _mm_and_ps(d, 0xFF7FFFFF);
	d = _mm_or_ps(d, 0x40000000);
	*r =_mm_add_ps(*r, _mm_div_ps(n,d));
}

inline int4 single_comupte(float4 n0, float4 n1, float4 n2, float4 n3, float cnt, float4 rnd_c, __local float4* sum)
{
	float4 c= (float4)(cnt);
	// 35 maths calls follow (140 FLOPS)
	float4 r = (float4)(0.0f);

	for(int i = 0; i < 4; ++i)
		round_compute(n0, n1, n2, n3, rnd_c, &c, &r);

	// do a quick fmod by setting exp to 2
	r = _mm_and_ps(r, 0x807FFFFF);
	r = _mm_or_ps(r, 0x40000000);
	*sum = r; // 34
	float4 x = (float4)(536870880.0f);
	r = _mm_mul_ps(r, x); // 35
	return convert_int4_rte(r);
}

inline void single_comupte_wrap(const uint rot, int4 v0, int4 v1, int4 v2, int4 v3, float cnt, float4 rnd_c, __local float4* sum, __local int4* out)
{
	float4 n0 = convert_float4_rte(v0);
	float4 n1 = convert_float4_rte(v1);
	float4 n2 = convert_float4_rte(v2);
	float4 n3 = convert_float4_rte(v3);

	int4 r = single_comupte(n0, n1, n2, n3, cnt, rnd_c, sum);
	*out = rot == 0 ? r : _mm_alignr_epi8(r, rot);
}

)==="
R"===(

static const __constant uint look[16][4] = {
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
	{3, 0, 2, 1}
};

static const __constant float ccnt[16] = {
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
	1.4609375f
};

__attribute__((reqd_work_group_size(WORKSIZE * 16, 1, 1)))
__kernel void JOIN(cn1_cn_gpu,ALGO)(__global int *lpad_in, __global int *spad, uint numThreads)
{
	const uint gIdx = getIdx();

#if(COMP_MODE==1)
	if(gIdx < Threads)
		return;
#endif

	uint chunk = get_local_id(0) / 16;

#if(STRIDED_INDEX==0)
	__global int* lpad = (__global int*)((__global char*)lpad_in + MEMORY * (gIdx/16));
#endif

	__local int4 smem2[1 * 4 * WORKSIZE];
	__local int4 smemOut2[1 * 16 * WORKSIZE];
	__local float4 smemVa2[1 * 16 * WORKSIZE];

	__local int4* smem = smem2 + 4 * chunk;
	__local int4* smemOut = smemOut2 + 16 * chunk;
	__local float4* smemVa = smemVa2 + 16 * chunk;

	uint tid = get_local_id(0) % 16;

	uint idxHash = gIdx/16;
	uint s = ((__global uint*)spad)[idxHash * 50] >> 8;
	float4 vs = (float4)(0);

	for(size_t i = 0; i < ITERATIONS; i++)
	{
		mem_fence(CLK_LOCAL_MEM_FENCE);
		((__local int*)smem)[tid] = ((__global int*)scratchpad_ptr(s, tid/4, lpad))[tid%4];
		mem_fence(CLK_LOCAL_MEM_FENCE);

		float4 rc = vs;

		{
			single_comupte_wrap(
				tid%4,
				*(smem + look[tid][0]),
				*(smem + look[tid][1]),
				*(smem + look[tid][2]),
				*(smem + look[tid][3]),
				ccnt[tid], rc, smemVa + tid,
				smemOut + tid
			);
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);

		int4 tmp2;
		if(tid % 4 == 0)
		{
			int4 out = _mm_xor_si128(smemOut[tid], smemOut[tid + 1]);
			int4 out2 = _mm_xor_si128(smemOut[tid + 2], smemOut[tid + 3]);
			out = _mm_xor_si128(out, out2);
			tmp2 = out;
			*scratchpad_ptr(s , tid/4, lpad) = _mm_xor_si128(smem[tid/4], out);
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);
		if(tid % 4 == 0)
			smemOut[tid] = tmp2;
		mem_fence(CLK_LOCAL_MEM_FENCE);
		int4 out2 = smemOut[0] ^ smemOut[4] ^ smemOut[8] ^ smemOut[12];

		if(tid%2 == 0)
			smemVa[tid] = smemVa[tid] + smemVa[tid + 1];
		if(tid%4 == 0)
			smemVa[tid] = smemVa[tid] + smemVa[tid + 2];
		if(tid%8 == 0)
			smemVa[tid] = smemVa[tid] + smemVa[tid + 4];
		if(tid%16 == 0)
			smemVa[tid] = smemVa[tid] + smemVa[tid + 8];
		vs = smemVa[0];

		vs = fabs(vs); // take abs(va) by masking the float sign bit
		float4 xx = _mm_mul_ps(vs, (float4)(16777216.0f));
		// vs range 0 - 64
		int4 tmp = convert_int4_rte(xx);
		tmp = _mm_xor_si128(tmp, out2);
		// vs is now between 0 and 1
		vs = _mm_div_ps(vs, (float4)(64.0f));
		s = tmp.x ^ tmp.y ^ tmp.z ^ tmp.w;
	}
}

)==="
R"===(

inline void generate_512(ulong idx, __local ulong* in, __global ulong* out)
{
	ulong hash[25];

	hash[0] = in[0] ^ idx;
	for(int i = 1; i < 25; ++i)
		hash[i] = in[i];

	keccakf1600_1(hash);
	for(int i = 0; i < 20; ++i)
		out[i] = hash[i];
	out+=160/8;

	keccakf1600_1(hash);
	for(int i = 0; i < 22; ++i)
		out[i] = hash[i];
	out+=176/8;

	keccakf1600_1(hash);
	for(int i = 0; i < 22; ++i)
		out[i] = hash[i];
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void JOIN(cn0_cn_gpu,ALGO)(__global ulong *input, __global int *Scratchpad, __global ulong *states, uint Threads)
{
    const uint gIdx = getIdx();
    __local ulong State_buf[8 * 25];
	__local ulong* State = State_buf + get_local_id(0) * 25;

#if(COMP_MODE==1)
    // do not use early return here
	if(gIdx < Threads)
#endif
    {
        states += 25 * gIdx;

#if(STRIDED_INDEX==0)
        Scratchpad = (__global int*)((__global char*)Scratchpad + MEMORY * gIdx);
#endif

        if (get_local_id(1) == 0)
        {

// NVIDIA
#ifdef __NV_CL_C_VERSION
			for(uint i = 0; i < 8; ++i)
				State[i] = input[i];
#else
            ((__local ulong8 *)State)[0] = vload8(0, input);
#endif
            State[8]  = input[8];
            State[9]  = input[9];
            State[10] = input[10];

            ((__local uint *)State)[9]  &= 0x00FFFFFFU;
            ((__local uint *)State)[9]  |= (((uint)get_global_id(0)) & 0xFF) << 24;
            ((__local uint *)State)[10] &= 0xFF000000U;
            /* explicit cast to `uint` is required because some OpenCL implementations (e.g. NVIDIA)
             * handle get_global_id and get_global_offset as signed long long int and add
             * 0xFFFFFFFF... to `get_global_id` if we set on host side a 32bit offset where the first bit is `1`
             * (even if it is correct casted to unsigned on the host)
             */
            ((__local uint *)State)[10] |= (((uint)get_global_id(0) >> 8));

            for (int i = 11; i < 25; ++i) {
                State[i] = 0x00UL;
            }

            // Last bit of padding
            State[16] = 0x8000000000000000UL;

            keccakf1600_2(State);

            #pragma unroll
            for (int i = 0; i < 25; ++i) {
                states[i] = State[i];
            }
        }
	}

	barrier(CLK_LOCAL_MEM_FENCE);

#if(COMP_MODE==1)
    // do not use early return here
	if(gIdx < Threads)
#endif
	{
		for(ulong i = get_local_id(1); i < MEMORY / 512; i += get_local_size(1))
		{
			generate_512(i, State, (__global ulong*)((__global uchar*)Scratchpad + i*512));
		}
	}
}

)==="
