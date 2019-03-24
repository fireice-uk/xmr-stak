
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

/** execute and check a CUDA api command
 *
 * @param id gpu id (thread id)
 * @param msg message string which should be added to the error message
 * @param ... CUDA api command
 */
#define CUDA_CHECK_MSG(id, msg, ...)                                                                          \
	{                                                                                                         \
		cudaError_t error = __VA_ARGS__;                                                                      \
		if(error != cudaSuccess)                                                                              \
		{                                                                                                     \
			std::cerr << "[CUDA] Error gpu " << id << ": <" << __FILE__ << ">:" << __LINE__;                  \
			std::cerr << msg << std::endl;                                                                    \
			throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error))); \
		}                                                                                                     \
	}                                                                                                         \
	((void)0)

#define CU_CHECK(id, ...)                                                                                                                                   \
	{                                                                                                                                                       \
		CUresult result = __VA_ARGS__;                                                                                                                      \
		if(result != CUDA_SUCCESS)                                                                                                                          \
		{                                                                                                                                                   \
			const char* s;                                                                                                                                  \
			cuGetErrorString(result, &s);                                                                                                                   \
			std::cerr << "[CUDA] Error gpu " << id << ": <" << __FUNCTION__ << ">:" << __LINE__ << " \"" << (s ? s : "unknown error") << "\"" << std::endl; \
			throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(s ? s : "unknown error"));                                                 \
		}                                                                                                                                                   \
	}                                                                                                                                                       \
	((void)0)

/** execute and check a CUDA api command
 *
 * @param id gpu id (thread id)
 * @param ... CUDA api command
 */
#define CUDA_CHECK(id, ...) CUDA_CHECK_MSG(id, "", __VA_ARGS__)

/** execute and check a CUDA kernel
 *
 * @param id gpu id (thread id)
 * @param ... CUDA kernel call
 */
#define CUDA_CHECK_KERNEL(id, ...) \
	__VA_ARGS__;                   \
	CUDA_CHECK(id, cudaGetLastError())

/** execute and check a CUDA kernel
 *
 * @param id gpu id (thread id)
 * @param msg message string which should be added to the error message
 * @param ... CUDA kernel call
 */
#define CUDA_CHECK_MSG_KERNEL(id, msg, ...) \
	__VA_ARGS__;                            \
	CUDA_CHECK_MSG(id, msg, cudaGetLastError())
