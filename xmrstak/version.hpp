#pragma once

#include <inttypes.h>
#include <string>

extern const char ver_long[];
extern const char ver_short[];
extern const char ver_html[];

inline std::string get_version_str()
{
	return std::string(ver_long);
}

inline std::string get_version_str_short()
{
	return std::string(ver_short);
}
