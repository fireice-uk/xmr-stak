
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>

/** execute and check a CUDA api command
 *
 * @param id gpu id (thread id)
 * @param ... CUDA api command
 */
#define CUDA_CHECK(id, ...) {														\
	cudaError_t error = __VA_ARGS__;											\
	if(error!=cudaSuccess){														\
		std::cerr << "[CUDA] Error gpu " << id << ": <" << __FILE__ << ">:" << __LINE__ << std::endl; \
		throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));	\
	}																			\
}																				\
( (void) 0 )

/** execute and check a CUDA kernel
 *
 * @param id gpu id (thread id)
 * @param ... CUDA kernel call
 */
#define CUDA_CHECK_KERNEL(id, ...)													\
	__VA_ARGS__;																\
	CUDA_CHECK(id, cudaGetLastError())
