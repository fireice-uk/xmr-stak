#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.hpp>
#include  <algorithm>
#include "xmrstak/jconf.hpp"

#ifdef __CUDACC__
__constant__
#else
const
#endif
uint64_t keccakf_rndc[24] ={
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

#include "cryptonight.hpp"
#include "cuda_extra.hpp"
#include "cuda_keccak.hpp"
#include "cuda_blake.hpp"
#include "cuda_groestl.hpp"
#include "cuda_jh.hpp"
#include "cuda_skein.hpp"
#include "cuda_device.hpp"

__constant__ uint8_t d_sub_byte[16][16] ={
	{0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76 },
	{0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0 },
	{0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15 },
	{0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75 },
	{0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84 },
	{0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf },
	{0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8 },
	{0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2 },
	{0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73 },
	{0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb },
	{0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79 },
	{0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08 },
	{0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a },
	{0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e },
	{0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf },
	{0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 }
};

__device__ __forceinline__ void cryptonight_aes_set_key( uint32_t * __restrict__ key, const uint32_t * __restrict__ data )
{
	int i, j;
	uint8_t temp[4];
	const uint32_t aes_gf[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

	MEMSET4( key, 0, 40 );
	MEMCPY4( key, data, 8 );

#pragma unroll
	for ( i = 8; i < 40; i++ )
	{
		*(uint32_t *) temp = key[i - 1];
		if ( i % 8 == 0 )
		{
			*(uint32_t *) temp = ROTR32( *(uint32_t *) temp, 8 );
			for ( j = 0; j < 4; j++ )
				temp[j] = d_sub_byte[( temp[j] >> 4 ) & 0x0f][temp[j] & 0x0f];
			*(uint32_t *) temp ^= aes_gf[i / 8 - 1];
		}
		else
		{
			if ( i % 8 == 4 )
			{
#pragma unroll
				for ( j = 0; j < 4; j++ )
					temp[j] = d_sub_byte[( temp[j] >> 4 ) & 0x0f][temp[j] & 0x0f];
			}
		}

		key[i] = key[( i - 8 )] ^ *(uint32_t *) temp;
	}
}

__global__ void cryptonight_extra_gpu_prepare( int threads, uint32_t * __restrict__ d_input, uint32_t len, uint32_t startNonce, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, uint32_t * __restrict__ d_ctx_key1, uint32_t * __restrict__ d_ctx_key2 )
{
	int thread = ( blockDim.x * blockIdx.x + threadIdx.x );

	if ( thread >= threads )
		return;

	uint32_t ctx_state[50];
	uint32_t ctx_a[4];
	uint32_t ctx_b[4];
	uint32_t ctx_key1[40];
	uint32_t ctx_key2[40];
	uint32_t input[21];

	memcpy( input, d_input, len );
	//*((uint32_t *)(((char *)input) + 39)) = startNonce + thread;
	uint32_t nonce = startNonce + thread;
	for ( int i = 0; i < sizeof (uint32_t ); ++i )
		( ( (char *) input ) + 39 )[i] = ( (char*) ( &nonce ) )[i]; //take care of pointer alignment

	cn_keccak( (uint8_t *) input, len, (uint8_t *) ctx_state );
	cryptonight_aes_set_key( ctx_key1, ctx_state );
	cryptonight_aes_set_key( ctx_key2, ctx_state + 8 );
	XOR_BLOCKS_DST( ctx_state, ctx_state + 8, ctx_a );
	XOR_BLOCKS_DST( ctx_state + 4, ctx_state + 12, ctx_b );

	memcpy( d_ctx_state + thread * 50, ctx_state, 50 * 4 );
	memcpy( d_ctx_a + thread * 4, ctx_a, 4 * 4 );
	memcpy( d_ctx_b + thread * 4, ctx_b, 4 * 4 );
	memcpy( d_ctx_key1 + thread * 40, ctx_key1, 40 * 4 );
	memcpy( d_ctx_key2 + thread * 40, ctx_key2, 40 * 4 );
}

__global__ void cryptonight_extra_gpu_final( int threads, uint64_t target, uint32_t* __restrict__ d_res_count, uint32_t * __restrict__ d_res_nonce, uint32_t * __restrict__ d_ctx_state )
{
	const int thread = blockDim.x * blockIdx.x + threadIdx.x;

	if ( thread >= threads )
		return;

	int i;
	uint32_t * __restrict__ ctx_state = d_ctx_state + thread * 50;
	uint64_t hash[4];
	uint32_t state[50];

#pragma unroll
	for ( i = 0; i < 50; i++ )
		state[i] = ctx_state[i];

	cn_keccakf2( (uint64_t *) state );

	switch ( ( (uint8_t *) state )[0] & 0x03 )
	{
	case 0:
		cn_blake( (const uint8_t *) state, 200, (uint8_t *) hash );
		break;
	case 1:
		cn_groestl( (const BitSequence *) state, 200, (BitSequence *) hash );
		break;
	case 2:
		cn_jh( (const BitSequence *) state, 200, (BitSequence *) hash );
		break;
	case 3:
		cn_skein( (const BitSequence *) state, 200, (BitSequence *) hash );
		break;
	default:
		break;
	}

	// Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
	// and expect an accurate result for target > 32-bit without implementing carries

	if ( hash[3] < target )
	{
		uint32_t idx = atomicInc( d_res_count, 0xFFFFFFFF );

		if(idx < 10)
			d_res_nonce[idx] = thread;
	}
}

extern "C" void cryptonight_extra_cpu_set_data( nvid_ctx* ctx, const void *data, uint32_t len )
{
	ctx->inputlen = len;
	CUDA_CHECK(ctx->device_id, cudaMemcpy( ctx->d_input, data, len, cudaMemcpyHostToDevice ));
}

extern "C" int cryptonight_extra_cpu_init(nvid_ctx* ctx)
{
	cudaError_t err;
	err = cudaSetDevice(ctx->device_id);
	if(err != cudaSuccess)
	{
		printf("GPU %d: %s", ctx->device_id, cudaGetErrorString(err));
		return 0;
	}

	cudaDeviceReset();
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	size_t hashMemSize;
	if(::jconf::inst()->IsCurrencyMonero())
	{
		hashMemSize = MONERO_MEMORY;
	}
	else
	{
		hashMemSize = AEON_MEMORY;
	}

	size_t wsize = ctx->device_blocks * ctx->device_threads;
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_long_state, hashMemSize * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_ctx_state, 50 * sizeof(uint32_t) * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_ctx_key1, 40 * sizeof(uint32_t) * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_ctx_key2, 40 * sizeof(uint32_t) * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_ctx_text, 32 * sizeof(uint32_t) * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_ctx_a, 4 * sizeof(uint32_t) * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_ctx_b, 4 * sizeof(uint32_t) * wsize));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_input, 21 * sizeof (uint32_t ) ));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_result_count, sizeof (uint32_t ) ));
	CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_result_nonce, 10 * sizeof (uint32_t ) ));
	return 1;
}

extern "C" void cryptonight_extra_cpu_prepare(nvid_ctx* ctx, uint32_t startNonce)
{
	int threadsperblock = 128;
	uint32_t wsize = ctx->device_blocks * ctx->device_threads;

	dim3 grid( ( wsize + threadsperblock - 1 ) / threadsperblock );
	dim3 block( threadsperblock );

	CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_extra_gpu_prepare<<<grid, block >>>( wsize, ctx->d_input, ctx->inputlen, startNonce,
		ctx->d_ctx_state, ctx->d_ctx_a, ctx->d_ctx_b, ctx->d_ctx_key1, ctx->d_ctx_key2 ));
}

extern "C" void cryptonight_extra_cpu_final(nvid_ctx* ctx, uint32_t startNonce, uint64_t target, uint32_t* rescount, uint32_t *resnonce)
{
	int threadsperblock = 128;
	uint32_t wsize = ctx->device_blocks * ctx->device_threads;

	dim3 grid( ( wsize + threadsperblock - 1 ) / threadsperblock );
	dim3 block( threadsperblock );

	CUDA_CHECK(ctx->device_id, cudaMemset( ctx->d_result_nonce, 0xFF, 10 * sizeof (uint32_t ) ));
	CUDA_CHECK(ctx->device_id, cudaMemset( ctx->d_result_count, 0, sizeof (uint32_t ) ));

	CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_extra_gpu_final<<<grid, block >>>( wsize, target, ctx->d_result_count, ctx->d_result_nonce, ctx->d_ctx_state ));

	CUDA_CHECK(ctx->device_id, cudaMemcpy( rescount, ctx->d_result_count, sizeof (uint32_t ), cudaMemcpyDeviceToHost ));
	CUDA_CHECK(ctx->device_id, cudaMemcpy( resnonce, ctx->d_result_nonce, 10 * sizeof (uint32_t ), cudaMemcpyDeviceToHost ));

	/* There is only a 32bit limit for the counter on the device side
	 * therefore this value can be greater than 10, in that case limit rescount
	 * to 10 entries.
	 */
	if(*rescount > 10)
		*rescount = 10;
	for(int i=0; i < *rescount; i++)
		resnonce[i] += startNonce;
}

extern "C" int cuda_get_devicecount( int* deviceCount)
{
	cudaError_t err;
	*deviceCount = 0;
	err = cudaGetDeviceCount(deviceCount);
	if(err != cudaSuccess)
	{
		if(err == cudaErrorNoDevice)
			printf("ERROR: NVIDIA no CUDA device found!\n");
		else if(err == cudaErrorInsufficientDriver)
			printf("WARNING: NVIDIA Insufficient driver!\n");
		else
			printf("WARNING: NVIDIA Unable to query number of CUDA devices!\n");
		return 0;
	}

	return 1;
}

/** get device information
 *
 * @return 0 = all OK,
 *         1 = something went wrong,
 *         2 = gpu cannot be selected,
 *         3 = context cannot be created
 *         4 = not enough memory
 *         5 = architecture not supported (not compiled for the gpu architecture)
 */
extern "C" int cuda_get_deviceinfo(nvid_ctx* ctx)
{
	cudaError_t err;
	int version;

	err = cudaDriverGetVersion(&version);
	if(err != cudaSuccess)
	{
		printf("Unable to query CUDA driver version! Is an nVidia driver installed?\n");
		return 1;
	}

	if(version < CUDART_VERSION)
	{
		printf("Driver does not support CUDA %d.%d API! Update your nVidia driver!\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
		return 1;
	}

	int GPU_N;
	if(cuda_get_devicecount(&GPU_N) == 0)
	{
		return 1;
	}

	if(ctx->device_id >= GPU_N)
	{
		printf("Invalid device ID!\n");
		return 1;
	}

	cudaDeviceProp props;
	err = cudaGetDeviceProperties(&props, ctx->device_id);
	if(err != cudaSuccess)
	{
		printf("\nGPU %d: %s\n%s line %d\n", ctx->device_id, cudaGetErrorString(err), __FILE__, __LINE__);
		return 1;
	}

	ctx->device_name = strdup(props.name);
	ctx->device_mpcount = props.multiProcessorCount;
	ctx->device_arch[0] = props.major;
	ctx->device_arch[1] = props.minor;

	const int gpuArch = ctx->device_arch[0] * 10 + ctx->device_arch[1];

	ctx->name = std::string(props.name);

	std::vector<int> arch;
#define XMRSTAK_PP_TOSTRING1(str) #str
#define XMRSTAK_PP_TOSTRING(str) XMRSTAK_PP_TOSTRING1(str)
	char const * archStringList = XMRSTAK_PP_TOSTRING(XMRSTAK_CUDA_ARCH_LIST);
#undef XMRSTAK_PP_TOSTRING
#undef XMRSTAK_PP_TOSTRING1
	std::stringstream ss(archStringList);

	//transform string list sperated with `+` into a vector of integers
	int tmpArch;
	while ( ss >> tmpArch )
		arch.push_back( tmpArch );

	if(gpuArch >= 20 && gpuArch < 30)
	{
		// compiled binary must support sm_20 for fermi
		std::vector<int>::iterator it = std::find(arch.begin(), arch.end(), 20);
		if(it == arch.end())
		{
			printf("WARNING: NVIDIA GPU %d: miner not compiled for the gpu architecture %d.\n", ctx->device_id, gpuArch);
			return 5;
		}
	}
	if(gpuArch >= 30)
	{
		// search the minimum architecture greater than sm_20
		int minSupportedArch = 0;
		/* - for newer architecture than fermi we need at least sm_30
		 * or a architecture >= gpuArch
		 * - it is not possible to use a gpu with a architecture >= 30
		 *   with a sm_20 only compiled binary
		 */
		for(int i = 0; i < arch.size(); ++i)
			if(minSupportedArch == 0 || (arch[i] >= 30 && arch[i] < minSupportedArch))
				minSupportedArch = arch[i];
		if(minSupportedArch < 30 || gpuArch < minSupportedArch)
		{
			printf("WARNING: NVIDIA GPU %d: miner not compiled for the gpu architecture %d.\n", ctx->device_id, gpuArch);
			return 5;
		}
	}

	// set all evice option those marked as auto (-1) to a valid value
	if(ctx->device_blocks == -1)
	{
		/* good values based of my experience
		 *	 - 3 * SMX count >=sm_30
		 *   - 2 * SMX count for <sm_30
		 */
		ctx->device_blocks = props.multiProcessorCount *
			( props.major < 3 ? 2 : 3 );
	}
	if(ctx->device_threads == -1)
	{
		/* sm_20 devices can only run 512 threads per cuda block
		 * `cryptonight_core_gpu_phase1` and `cryptonight_core_gpu_phase3` starts
		 * `8 * ctx->device_threads` threads per block
		 */
		ctx->device_threads = 64;
		constexpr size_t byteToMiB = 1024u * 1024u;
		
		// no limit by default 1TiB
		size_t maxMemUsage = byteToMiB * byteToMiB;
		if(props.major < 6)
		{
			// limit memory usage for GPUs before pascal
			maxMemUsage = size_t(2048u) * byteToMiB;
		}
		if(props.major == 2)
		{
			// limit memory usage for sm 20 GPUs
			maxMemUsage = size_t(1024u) * byteToMiB;
		}

		int* tmp;
		cudaError_t err;
		// a device must be selected to get the right memory usage later on
		err = cudaSetDevice(ctx->device_id);
		if(err != cudaSuccess)
		{
			printf("WARNING: NVIDIA GPU %d: cannot be selected.\n", ctx->device_id);
			return 2;
		}
		// trigger that a context on the gpu will be allocated
		err = cudaMalloc(&tmp, 256);
		if(err != cudaSuccess)
		{
			printf("WARNING: NVIDIA GPU %d: context cannot be created.\n", ctx->device_id);
			return 3;
		}


		size_t freeMemory = 0;
		size_t totalMemory = 0;
		CUDA_CHECK(ctx->device_id, cudaMemGetInfo(&freeMemory, &totalMemory));

		cudaFree(tmp);
		// delete created context on the gpu
		cudaDeviceReset();
		
		ctx->total_device_memory = totalMemory;
		ctx->free_device_memory = freeMemory;

		size_t hashMemSize;
		if(::jconf::inst()->IsCurrencyMonero())
		{
			hashMemSize = MONERO_MEMORY;
		}
		else
		{
			hashMemSize = AEON_MEMORY;
		}

#ifdef WIN32
		/* We use in windows bfactor (split slow kernel into smaller parts) to avoid
		 * that windows is killing long running kernel.
		 * In the case there is already memory used on the gpu than we
		 * assume that other application are running between the split kernel,
		 * this can result into TLB memory flushes and can strongly reduce the performance
		 * and the result can be that windows is killing the miner.
		 * Be reducing maxMemUsage we try to avoid this effect.
		 */
		size_t usedMem = totalMemory - freeMemory;
		if(usedMem >= maxMemUsage)
		{
			printf("WARNING: NVIDIA GPU %d: already %s MiB memory in use, skip GPU.\n",
				ctx->device_id,
				std::to_string(usedMem/byteToMiB).c_str());
			return 4;
		}
		else
			maxMemUsage -= usedMem;

#endif
		// keep 128MiB memory free (value is randomly chosen)
		// 200byte are meta data memory (result nonce, ...)
		size_t availableMem = freeMemory - (128u * byteToMiB) - 200u;
		size_t limitedMemory = std::min(availableMem, maxMemUsage);
		// up to 16kibyte extra memory is used per thread for some kernel (lmem/local memory)
		// 680bytes are extra meta data memory per hash
		size_t perThread = hashMemSize + 16192u + 680u;
		size_t max_intensity = limitedMemory / perThread;
		ctx->device_threads = max_intensity / ctx->device_blocks;
		// use only odd number of threads
		ctx->device_threads = ctx->device_threads & 0xFFFFFFFE;

		if(props.major == 2 && ctx->device_threads > 64)
		{
			// Fermi gpus only support 512 threads per block (we need start 4 * configured threads)
			ctx->device_threads = 64;
		}

	}

	return 0;
}
