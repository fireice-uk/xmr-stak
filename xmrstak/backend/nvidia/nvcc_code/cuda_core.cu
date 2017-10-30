#include "xmrstak/backend/cryptonight.hpp"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
    if (waitTime > 0)
    {
        if (waitTime > 100)
        {
            // use a waitable timer for larger intervals > 0.1ms

            HANDLE timer;
            LARGE_INTEGER ft;

            ft.QuadPart = -(10*waitTime); // Convert to 100 nanosecond interval, negative value indicates relative time

            timer = CreateWaitableTimer(NULL, TRUE, NULL);
            SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
            WaitForSingleObject(timer, INFINITE);
            CloseHandle(timer);
        }
        else
        {
            // use a polling loop for short intervals <= 100ms

            LARGE_INTEGER perfCnt, start, now;
            __int64 elapsed;

            QueryPerformanceFrequency(&perfCnt);
            QueryPerformanceCounter(&start);
            do {
		SwitchToThread();
                QueryPerformanceCounter((LARGE_INTEGER*) &now);
                elapsed = (__int64)((now.QuadPart - start.QuadPart) / (float)perfCnt.QuadPart * 1000 * 1000);
            } while ( elapsed < waitTime );
        }
    }
}
#else
#include <unistd.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
	usleep(waitTime);
}
#endif

#include "cryptonight.hpp"
#include "cuda_extra.hpp"
#include "cuda_aes.hpp"
#include "cuda_device.hpp"

/* sm_2X is limited to 2GB due to the small TLB
 * therefore we never use 64bit indices
 */
#if defined(XMR_STAK_LARGEGRID) && (__CUDA_ARCH__ >= 300)
typedef uint64_t IndexType;
#else
typedef int IndexType;
#endif

__device__ __forceinline__ uint64_t cuda_mul128( uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi )
{
	*product_hi = __umul64hi( multiplier, multiplicand );
	return (multiplier * multiplicand );
}

template< typename T >
__device__ __forceinline__ T loadGlobal64( T * const addr )
{
	T x;
	asm volatile( "ld.global.cg.u64 %0, [%1];" : "=l"( x ) : "l"( addr ) );
	return x;
}

template< typename T >
__device__ __forceinline__ T loadGlobal32( T * const addr )
{
	T x;
	asm volatile( "ld.global.cg.u32 %0, [%1];" : "=r"( x ) : "l"( addr ) );
	return x;
}


template< typename T >
__device__ __forceinline__ void storeGlobal32( T* addr, T const & val )
{
	asm volatile( "st.global.cg.u32 [%0], %1;" : : "l"( addr ), "r"( val ) );
}

template<size_t ITERATIONS, uint32_t THREAD_SHIFT>
__global__ void cryptonight_core_gpu_phase1( int threads, int bfactor, int partidx, uint32_t * __restrict__ long_state, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	const int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 3;
	const int sub = ( threadIdx.x & 7 ) << 2;

	const int batchsize = ITERATIONS >> bfactor;
	const int start = partidx * batchsize;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40], text[4];

	MEMCPY8( key, ctx_key1 + thread * 40, 20 );

	if( partidx == 0 )
	{
		// first round
		MEMCPY8( text, ctx_state + thread * 50 + sub + 16, 2 );
	}
	else
	{
		// load previous text data
		MEMCPY8( text, &long_state[( (uint64_t) thread << THREAD_SHIFT ) + sub + start - 32], 2 );
	}
	__syncthreads( );
	for ( int i = start; i < end; i += 32 )
	{
		cn_aes_pseudo_round_mut( sharedMemory, text, key );
		MEMCPY8(&long_state[((uint64_t) thread << THREAD_SHIFT) + (sub + i)], text, 2);
	}
}

/** avoid warning `unused parameter` */
template< typename T >
__forceinline__ __device__ void unusedVar( const T& )
{
}

/** shuffle data for
 *
 * - this method can be used with all compute architectures
 * - for <sm_30 shared memory is needed
 *
 * @param ptr pointer to shared memory, size must be `threadIdx.x * sizeof(uint32_t)`
 *            value can be NULL for compute architecture >=sm_30
 * @param sub thread number within the group, range [0;4)
 * @param value value to share with other threads within the group
 * @param src thread number within the group from where the data is read, range [0;4)
 */
__forceinline__ __device__ uint32_t shuffle(volatile uint32_t* ptr,const uint32_t sub,const int val,const uint32_t src)
{
#if( __CUDA_ARCH__ < 300 )
    ptr[sub] = val;
    return ptr[src&3];
#else
    unusedVar( ptr );
    unusedVar( sub );
#   if(__CUDACC_VER_MAJOR__ >= 9)
    return __shfl_sync(0xFFFFFFFF, val, src, 4 );
#	else
	return __shfl( val, src, 4 );
#	endif
#endif
}

#ifdef XMR_STAK_THREADS
__launch_bounds__( XMR_STAK_THREADS * 4 )
#endif
template<size_t ITERATIONS, uint32_t THREAD_SHIFT, uint32_t MASK>
__global__ void cryptonight_core_gpu_phase2( int threads, int bfactor, int partidx, uint32_t * d_long_state, uint32_t * d_ctx_a, uint32_t * d_ctx_b )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );

	__syncthreads( );

	const int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 2;
	const int sub = threadIdx.x & 3;
	const int sub2 = sub & 2;

#if( __CUDA_ARCH__ < 300 )
        extern __shared__ uint32_t shuffleMem[];
        volatile uint32_t* sPtr = (volatile uint32_t*)(shuffleMem + (threadIdx.x& 0xFFFFFFFC));
#else
        volatile uint32_t* sPtr = NULL;
#endif
	if ( thread >= threads )
		return;

	int i, k;
        uint32_t j;
	const int batchsize = (ITERATIONS * 2) >> ( 2 + bfactor );
	const int start = partidx * batchsize;
	const int end = start + batchsize;
	uint32_t * long_state = &d_long_state[(IndexType) thread << THREAD_SHIFT];
	uint32_t * ctx_a = d_ctx_a + thread * 4;
	uint32_t * ctx_b = d_ctx_b + thread * 4;
	uint32_t a, d[2];
	uint32_t t1[2], t2[2], res;

	a = ctx_a[sub];
	d[1] = ctx_b[sub];
	#pragma unroll 2
	for ( i = start; i < end; ++i )
	{
		#pragma unroll 2
		for ( int x = 0; x < 2; ++x )
		{
			j = ( ( shuffle(sPtr,sub, a, 0) & MASK ) >> 2 ) + sub;

			const uint32_t x_0 = loadGlobal32<uint32_t>( long_state + j );
			const uint32_t x_1 = shuffle(sPtr,sub, x_0, sub + 1);
			const uint32_t x_2 = shuffle(sPtr,sub, x_0, sub + 2);
			const uint32_t x_3 = shuffle(sPtr,sub, x_0, sub + 3);
			d[x] = a ^
				t_fn0( x_0 & 0xff ) ^
				t_fn1( (x_1 >> 8) & 0xff ) ^
				t_fn2( (x_2 >> 16) & 0xff ) ^
				t_fn3( ( x_3 >> 24 ) );


			//XOR_BLOCKS_DST(c, b, &long_state[j]);
			t1[0] = shuffle(sPtr,sub, d[x], 0);
			//long_state[j] = d[0] ^ d[1];
			storeGlobal32( long_state + j, d[0] ^ d[1] );

			//MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & MASK]);
			j = ( ( *t1 & MASK ) >> 2 ) + sub;

			uint32_t yy[2];
			*( (uint64_t*) yy ) = loadGlobal64<uint64_t>( ( (uint64_t *) long_state )+( j >> 1 ) );
			uint32_t zz[2];
			zz[0] = shuffle(sPtr,sub, yy[0], 0);
			zz[1] = shuffle(sPtr,sub, yy[1], 0);

			t1[1] = shuffle(sPtr,sub, d[x], 1);
			#pragma unroll
			for ( k = 0; k < 2; k++ )
				t2[k] = shuffle(sPtr,sub, a, k + sub2);

            *( (uint64_t *) t2 ) += sub2 ? ( *( (uint64_t *) t1 ) * *( (uint64_t*) zz ) ) : __umul64hi( *( (uint64_t *) t1 ), *( (uint64_t*) zz ) );

			res = *( (uint64_t *) t2 )  >> ( sub & 1 ? 32 : 0 );

			storeGlobal32( long_state + j, res );
			a = ( sub & 1 ? yy[1] : yy[0] ) ^ res;
		}
	}

	if ( bfactor > 0 )
	{
		ctx_a[sub] = a;
		ctx_b[sub] = d[1];
	}
}

template<size_t ITERATIONS, uint32_t THREAD_SHIFT>
__global__ void cryptonight_core_gpu_phase3( int threads, int bfactor, int partidx, const uint32_t * __restrict__ long_state, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 3;
	int sub = ( threadIdx.x & 7 ) << 2;

	const int batchsize = ITERATIONS >> bfactor;
	const int start = partidx * batchsize;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40], text[4];
	MEMCPY8( key, d_ctx_key2 + thread * 40, 20 );
	MEMCPY8( text, d_ctx_state + thread * 50 + sub + 16, 2 );

	__syncthreads( );
	for ( int i = start; i < end; i += 32 )
	{
#pragma unroll
		for ( int j = 0; j < 4; ++j )
			text[j] ^= long_state[((IndexType) thread << THREAD_SHIFT) + (sub + i + j)];

		cn_aes_pseudo_round_mut( sharedMemory, text, key );
	}

	MEMCPY8( d_ctx_state + thread * 50 + sub + 16, text, 2 );
}

template<size_t ITERATIONS, uint32_t MASK, uint32_t THREAD_SHIFT>
void cryptonight_core_gpu_hash(nvid_ctx* ctx)
{
	dim3 grid( ctx->device_blocks );
	dim3 block( ctx->device_threads );
	dim3 block4( ctx->device_threads << 2 );
	dim3 block8( ctx->device_threads << 3 );

	int partcount = 1 << ctx->device_bfactor;

	/* bfactor for phase 1 and 3
	 *
	 * phase 1 and 3 consume less time than phase 2, therefore we begin with the
	 * kernel splitting if the user defined a `bfactor >= 5`
	 */
	int bfactorOneThree = ctx->device_bfactor - 4;
	if( bfactorOneThree < 0 )
		bfactorOneThree = 0;

	int partcountOneThree = 1 << bfactorOneThree;

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase1<ITERATIONS,THREAD_SHIFT><<< grid, block8 >>>( ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state, ctx->d_ctx_state, ctx->d_ctx_key1 ));

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}
	if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );

	for ( int i = 0; i < partcount; i++ )
	{
        CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase2<ITERATIONS,THREAD_SHIFT,MASK><<<
            grid,
            block4,
            block4.x * sizeof(uint32_t) * static_cast< int >( ctx->device_arch[0] < 3 )
        >>>(
            ctx->device_blocks*ctx->device_threads,
            ctx->device_bfactor,
            i,
            ctx->d_long_state,
            ctx->d_ctx_a,
            ctx->d_ctx_b
        ));

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase3<ITERATIONS,THREAD_SHIFT><<< grid, block8 >>>( ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state,
			ctx->d_ctx_state, ctx->d_ctx_key2 ));
	}
}

void cryptonight_core_cpu_hash(nvid_ctx* ctx, bool mineMonero)
{
#ifndef CONF_NO_MONERO
	if(mineMonero)
	{
		cryptonight_core_gpu_hash<MONERO_ITER, MONERO_MASK, 19u>(ctx);
	}
#endif
#ifndef CONF_NO_AEON
	if(!mineMonero)
	{
		cryptonight_core_gpu_hash<AEON_ITER, AEON_MASK, 18u>(ctx);
	}
#endif
}
