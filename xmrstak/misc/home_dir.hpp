#pragma once

#include <string>

#ifdef _WIN32
#include <WinSock2.h>
#include <Shlobj.h>

namespace
{
	inline std::string get_home()
	{
		char path[MAX_PATH + 1];
		// get folder "appdata\local"
		if (SHGetSpecialFolderPathA(HWND_DESKTOP, path, CSIDL_LOCAL_APPDATA, FALSE))
		{
			return path;
		}
		else
			return ".";
	}
} // namespace anonymous

#else
#include <unistd.h>
#include <pwd.h>
#include <cstdlib>

namespace
{
	inline std::string get_home()
	{
		const char *home = ".";

		if ((home = getenv("HOME")) == nullptr)
			home = getpwuid(getuid())->pw_dir;

		return home;
	}
} // namespace anonymous

#endif // _WIN32
