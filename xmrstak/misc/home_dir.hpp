#pragma once

#include <string>

#ifdef _WIN32
#include <WinSock2.h>
// this comment avoid that clang format reorders the includes
#include <Shlobj.h>

namespace
{
inline std::string get_home()
{
	char path[MAX_PATH + 1];
	// get folder "appdata\local"
	if(SHGetSpecialFolderPathA(HWND_DESKTOP, path, CSIDL_LOCAL_APPDATA, FALSE))
	{
		return path;
	}
	else
		return ".";
}
} // namespace

#else
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>

namespace
{
inline std::string get_home()
{
	const char* home = ".";

	if((home = getenv("HOME")) == nullptr)
		home = getpwuid(getuid())->pw_dir;

	return home;
}
} // namespace

#endif // _WIN32
