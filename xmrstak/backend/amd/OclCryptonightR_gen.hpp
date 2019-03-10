#pragma once

#include "xmrstak/backend/cryptonight.hpp"

#include <stdint.h>
#include <vector>
#include <string>

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "xmrstak/backend/amd/amd_gpu/gpu.hpp"

namespace xmrstak
{
namespace amd
{

cl_program CryptonightR_get_program(GpuContext* ctx, const xmrstak_algo algo,
	uint64_t height, uint32_t precompile_count, bool background = false, cl_kernel old_kernel = nullptr);

} // namespace amd
} // namespace xmrstak
