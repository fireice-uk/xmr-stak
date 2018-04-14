#pragma once

#include <inttypes.h>
#include <string>
#include "donate-level.hpp"

#define XMR_STAK_NAME "xmr-stak"
#define XMR_STAK_VERSION "2.4.2"

extern const char ver_long[];
extern const char ver_short[];
extern const char ver_html[];

inline std::string get_version_str()
{
	return std::string(ver_long) + std::to_string(uint32_t(fDevDonationLevel * 1000)) ;
}

inline std::string get_version_str_short()
{
	return std::string(ver_short);
}
