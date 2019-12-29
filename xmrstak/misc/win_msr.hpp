#pragma once

#include <inttypes.h>
#include <vector>

struct msr_reg
{
	uint32_t addr;
	uint64_t val;
};

#ifdef _WIN32
void load_win_msrs(const std::vector<msr_reg>& regs);
#else
void load_win_msrs(const std::vector<msr_reg>& regs) {}
#endif

