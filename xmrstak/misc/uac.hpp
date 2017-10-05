#pragma once
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

BOOL SelfElevate(const char* my_path)
{
	if (IsElevated())
		return FALSE;

	SHELLEXECUTEINFO shExecInfo = { 0 };
	shExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	shExecInfo.fMask = NULL;
	shExecInfo.hwnd = NULL;
	shExecInfo.lpVerb = "runas";
	shExecInfo.lpFile = my_path;
	shExecInfo.lpParameters = NULL;
	shExecInfo.lpDirectory = NULL;
	shExecInfo.nShow = SW_SHOW;
	shExecInfo.hInstApp = NULL;

	if (!ShellExecuteEx(&shExecInfo))
		return FALSE;

	// Hide our window and loiter in the background to make scripting easier
	// ShowWindow(GetConsoleWindow(), SW_HIDE);
	// WaitForSingleObject(shExecInfo.hProcess, INFINITE);

	return TRUE;
}
