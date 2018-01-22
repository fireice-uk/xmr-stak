#pragma once

#ifdef _WIN32
#include <string>

BOOL IsElevated();
BOOL SelfElevate(const std::string& my_path, const std::string& params);
VOID RequestElevation();
BOOL IsWindows10OrNewer();
#endif
