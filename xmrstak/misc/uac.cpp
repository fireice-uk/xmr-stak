#ifdef _WIN32
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/jconf.hpp"

#include <string>
#include <windows.h>

BOOL IsElevated()
{
	BOOL fRet = FALSE;
	HANDLE hToken = NULL;
	if(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken))
	{
		TOKEN_ELEVATION Elevation;
		DWORD cbSize = sizeof(TOKEN_ELEVATION);
		if(GetTokenInformation(hToken, TokenElevation, &Elevation, sizeof(Elevation), &cbSize))
			fRet = Elevation.TokenIsElevated;
	}
	if(hToken)
		CloseHandle(hToken);
	return fRet;
}

BOOL SelfElevate(const std::string& my_path, const std::string& params)
{
	if(IsElevated())
		return FALSE;

	SHELLEXECUTEINFO shExecInfo = {0};
	shExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	shExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
	shExecInfo.hwnd = NULL;
	shExecInfo.lpVerb = "runas";
	shExecInfo.lpFile = my_path.c_str();
	shExecInfo.lpParameters = params.c_str();
	shExecInfo.lpDirectory = NULL;
	shExecInfo.nShow = SW_SHOW;
	shExecInfo.hInstApp = NULL;

	if(!ShellExecuteEx(&shExecInfo))
		return FALSE;

	// Loiter in the background to make scripting easier
	printer::inst()->print_msg(L0, "This window has been opened because xmr-stak needed to run as administrator.  It can be safely closed now.");
	WaitForSingleObject(shExecInfo.hProcess, INFINITE);
	std::exit(0);

	return TRUE;
}

VOID RequestElevation()
{
	if(IsElevated())
		return;

	if(!xmrstak::params::inst().allowUAC)
	{
		printer::inst()->print_msg(L0, "The miner needs to run as administrator, but you passed --noUAC option. Please remove it or set use_slow_memory to always.");
		if (::jconf::inst()->GetSlowMemSetting() == ::jconf::print_warning)
			return;
		
		win_exit();
		return;
	}

	SelfElevate(xmrstak::params::inst().minerArg0, xmrstak::params::inst().minerArgs);
}

BOOL IsWindows10OrNewer()
{
	OSVERSIONINFOEX osvi = {0};
	osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
	osvi.dwMajorVersion = 10;
	osvi.dwMinorVersion = 0;
	DWORDLONG dwlConditionMask = 0;
	VER_SET_CONDITION(dwlConditionMask, VER_MAJORVERSION, VER_GREATER_EQUAL);
	VER_SET_CONDITION(dwlConditionMask, VER_MINORVERSION, VER_GREATER_EQUAL);
	return ::VerifyVersionInfo(&osvi, VER_MAJORVERSION | VER_MINORVERSION, dwlConditionMask);
}
#endif
