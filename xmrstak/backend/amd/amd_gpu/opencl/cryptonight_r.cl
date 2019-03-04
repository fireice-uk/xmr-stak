R"===(
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

#define cryptonight_r_wow 15
#define cryptonight_r 16

#define MEM_CHUNK (1 << MEM_CHUNK_EXPONENT)

#if(STRIDED_INDEX==0)
#   define IDX(x)	(x)
#elif(STRIDED_INDEX==1)
#	define IDX(x)   (mul24(((uint)(x)), Threads))
#elif(STRIDED_INDEX==2)
#   define IDX(x)	(((x) % MEM_CHUNK) + ((x) / MEM_CHUNK) * WORKSIZE * MEM_CHUNK)
#elif(STRIDED_INDEX==3)
#	define IDX(x)   ((x) * WORKSIZE)
#endif

// __NV_CL_C_VERSION checks if NVIDIA opencl is used
#if(ALGO == cryptonight_monero_v8 && defined(__NV_CL_C_VERSION))
#	define SCRATCHPAD_CHUNK(N) (*(__local uint4*)((__local uchar*)(scratchpad_line) + (idx1 ^ (N << 4))))
#	define SCRATCHPAD_CHUNK_GLOBAL (*((__global uint16*)(Scratchpad + (IDX((idx0 & 0x1FFFC0U) >> 4)))))
#else
#	define SCRATCHPAD_CHUNK(N) (Scratchpad[IDX(((idx) >> 4) ^ N)])
#endif

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_cryptonight_r(__global uint4 *Scratchpad, __global ulong *states, uint Threads)
{
    ulong a[2], b[4];
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];

#ifdef __NV_CL_C_VERSION
	__local uint16 scratchpad_line_buf[WORKSIZE];
 	__local uint16* scratchpad_line = scratchpad_line_buf + get_local_id(0);
#endif

    const ulong gIdx = get_global_id(0) - get_global_offset(0);

    for(int i = get_local_id(0); i < 256; i += WORKSIZE)
    {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#if(STRIDED_INDEX==0)
		Scratchpad += gIdx * (MEMORY >> 4);
#elif(STRIDED_INDEX==1)
		Scratchpad += gIdx;
#elif(STRIDED_INDEX==2)
		Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#elif(STRIDED_INDEX==3)
		Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + (gIdx % WORKSIZE);
#endif

        a[0] = states[0] ^ states[4];
        a[1] = states[1] ^ states[5];

        b[0] = states[2] ^ states[6];
        b[1] = states[3] ^ states[7];
        b[2] = states[8] ^ states[10];
        b[3] = states[9] ^ states[11];
    }

    ulong2 bx0 = ((ulong2 *)b)[0];
    ulong2 bx1 = ((ulong2 *)b)[1];

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {

	uint r0 = as_uint2(states[12]).s0;
	uint r1 = as_uint2(states[12]).s1;
	uint r2 = as_uint2(states[13]).s0;
	uint r3 = as_uint2(states[13]).s1;

    #pragma unroll CN_UNROLL
    for(int i = 0; i < ITERATIONS; ++i)
    {
#       ifdef __NV_CL_C_VERSION
            uint idx  = a[0] & 0x1FFFC0;
            uint idx1 = a[0] & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            uint idx = a[0] & MASK;
#       endif

#if(ALGO == cryptonight_monero_v8 && defined(__NV_CL_C_VERSION))
 		*scratchpad_line = SCRATCHPAD_CHUNK_GLOBAL;
#endif
        uint4 c = SCRATCHPAD_CHUNK(0);
        c = AES_Round(AES0, AES1, AES2, AES3, c, ((uint4 *)a)[0]);

        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

#if (ALGO == cryptonight_r)
            c ^= as_uint4(chunk1) ^ as_uint4(chunk2) ^ as_uint4(chunk3);
#endif

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

        SCRATCHPAD_CHUNK(0) = as_uint4(bx0) ^ c;

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;

            idx = as_ulong2(c).s0 & 0x1FFFC0;
            idx1 = as_ulong2(c).s0 & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            idx = as_ulong2(c).s0 & MASK;
#       endif

        uint4 tmp = SCRATCHPAD_CHUNK(0);

        tmp.s0 ^= r0 + r1;
        tmp.s1 ^= r2 + r3;
        const uint r4 = as_uint2(a[0]).s0;
        const uint r5 = as_uint2(a[1]).s0;
        const uint r6 = as_uint4(bx0).s0;
        const uint r7 = as_uint4(bx1).s0;
#if (ALGO == cryptonight_r)
        const uint r8 = as_uint4(bx1).s2;
#endif
#define ROT_BITS 32

	XMRSTAK_INCLUDE_RANDOM_MATH

#if (ALGO == cryptonight_r)

        const uint2 al = (uint2)(as_uint2(a[0]).s0 ^ r2, as_uint2(a[0]).s1 ^ r3);
        const uint2 ah = (uint2)(as_uint2(a[1]).s0 ^ r0, as_uint2(a[1]).s1 ^ r1);
#endif

        ulong2 t;
        t.s0 = mul_hi(as_ulong2(c).s0, as_ulong2(tmp).s0);
        t.s1 = as_ulong2(c).s0 * as_ulong2(tmp).s0;
        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1))
#if (ALGO == cryptonight_r_wow)
            ^ t
#endif
            ;
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
#if (ALGO == cryptonight_r_wow)
            t ^= chunk2;
#endif
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

#if (ALGO == cryptonight_r)
            c ^= as_uint4(chunk1) ^ as_uint4(chunk2) ^ as_uint4(chunk3);
#endif

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

#if (ALGO == cryptonight_r)
        a[1] = as_ulong(ah) + t.s1;
        a[0] = as_ulong(al) + t.s0;
#else
        a[1] += t.s1;
        a[0] += t.s0;
#endif

        SCRATCHPAD_CHUNK(0) = ((uint4 *)a)[0];

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;
#       endif

        ((uint4 *)a)[0] ^= tmp;
        bx1 = bx0;
        bx0 = as_ulong2(c);
    }

#   undef SCRATCHPAD_CHUNK
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}
)==="
