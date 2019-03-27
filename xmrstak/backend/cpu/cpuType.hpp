#pragma once

#include <cstdint>
#include <string>

namespace xmrstak
{
namespace cpu
{
struct Model
{
	uint32_t family = 0u;
	uint32_t model = 0u;
	bool isIntelXBridge = false;
	bool isIntelXWell = false;
	bool isIntelXLake = false;
	bool isAMDHammer = false;
	bool isAMDBulldozer = false;
	bool isAMDZen = false;
	bool aes = false;
	bool sse2 = false;
	bool sse3 = false;
	bool ssse3 = false;
	bool avx = false;
	bool avx2 = false;
	std::string type_name = "unknown";
	std::string brand_name = "unknown";
};

Model getModel();

/** Mask bits between h and l and return the value
	 *
	 * This enables us to put in values exactly like in the manual
	 * For example EBX[30:22] is get_masked(cpu_info[1], 31, 22)
	 */
uint32_t get_masked(int32_t val, int32_t h, int32_t l);


} // namespace cpu
} // namespace xmrstak
