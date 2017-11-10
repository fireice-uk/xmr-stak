#pragma once

#include <string>
#include "donate-level.hpp"

extern const char ver_long[];
extern const char ver_short[];

inline std::string get_version_str()
{
	return std::string(ver_long) + std::to_string(uint(fDevDonationLevel * 1000)) ;
}

inline std::string get_version_str_short()
{
	return std::string(ver_short);
}
