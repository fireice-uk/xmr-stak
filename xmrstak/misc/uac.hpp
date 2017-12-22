#pragma once

#ifdef _WIN32
#include "xmrstak/misc/console.hpp"

#include <string>
#include <windows.h>

BOOL IsElevated() 
{
	BOOL fRet = FALSE;
	HANDLE hToken = NULL;
	if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) 
	{
		TOKEN_ELEVATION Elevation;
		DWORD cbSize = sizeof(TOKEN_ELEVATION);
		if (GetTokenInformation(hToken, TokenElevation, &Elevation, sizeof(Elevation), &cbSize))
			fRet = Elevation.TokenIsElevated;
	}
	if (hToken)
		CloseHandle(hToken);
	return fRet;
}

BOOL SelfElevate(const char* my_path, const std::string& params)
{
	if (IsElevated())
		return FALSE;

	SHELLEXECUTEINFO shExecInfo = { 0 };
	shExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	shExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
	shExecInfo.hwnd = NULL;
	shExecInfo.lpVerb = "runas";
	shExecInfo.lpFile = my_path;
	shExecInfo.lpParameters = params.c_str();
	shExecInfo.lpDirectory = NULL;
	shExecInfo.nShow = SW_SHOW;
	shExecInfo.hInstApp = NULL;

	if (!ShellExecuteEx(&shExecInfo))
		return FALSE;

	// Loiter in the background to make scripting easier
	printer::inst()->print_msg(L0, "This window has been opened because xmr-stak needed to run as administrator.  It can be safely closed now.");
	WaitForSingleObject(shExecInfo.hProcess, INFINITE);
	std::exit(0);

	return TRUE;
}
#endif
