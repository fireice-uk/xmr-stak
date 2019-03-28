#pragma once

#include "xmrstak/misc/environment.hpp"
#include "xmrstak/params.hpp"

#include "iBackend.hpp"
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifndef USE_PRECOMPILED_HEADERS
#ifdef WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/types.h>
#endif
#include <iostream>
#endif

namespace xmrstak
{

struct plugin
{

	plugin() = default;

	void load(const std::string backendName, const std::string libName)
	{
		m_backendName = backendName;
#ifdef WIN32
		libBackend = LoadLibrary(TEXT((libName + ".dll").c_str()));
		if(!libBackend)
		{
			std::cerr << "WARNING: " << m_backendName << " cannot load backend library: " << (libName + ".dll") << std::endl;
			return;
		}
#else
		// `.so` linux file extention for dynamic libraries
		std::string fileExtension = ".so";
#if defined(__APPLE__)
		// `.dylib` Mac OS X file extention for dynamic libraries
		fileExtension = ".dylib";
#endif
		// search library in working directory
		libBackend = dlopen(("./lib" + libName + fileExtension).c_str(), RTLD_LAZY);
		// fallback to binary directory
		if(!libBackend)
			libBackend = dlopen((params::inst().executablePrefix + "lib" + libName + fileExtension).c_str(), RTLD_LAZY);
		// try use LD_LIBRARY_PATH
		if(!libBackend)
			libBackend = dlopen(("lib" + libName + fileExtension).c_str(), RTLD_LAZY);
		if(!libBackend)
		{
			std::cerr << "WARNING: " << m_backendName << " cannot load backend library: " << dlerror() << std::endl;
			return;
		}
#endif

#ifdef WIN32
		fn_startBackend = (startBackend_t)GetProcAddress(libBackend, "xmrstak_start_backend");
		if(!fn_startBackend)
		{
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_start_backend': " << GetLastError() << std::endl;
		}
#else
		// reset last error
		dlerror();
		fn_startBackend = (startBackend_t)dlsym(libBackend, "xmrstak_start_backend");
		const char* dlsym_error = dlerror();
		if(dlsym_error)
		{
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_start_backend': " << dlsym_error << std::endl;
		}
#endif
	}

	std::vector<iBackend*>* startBackend(uint32_t threadOffset, miner_work& pWork, environment& env)
	{
		if(fn_startBackend == nullptr)
		{
			std::vector<iBackend*>* pvThreads = new std::vector<iBackend*>();
			return pvThreads;
		}

		return fn_startBackend(threadOffset, pWork, env);
	}

	void unload()
	{
		if(libBackend)
		{
#ifdef WIN32
			FreeLibrary(libBackend);
#else
			dlclose(libBackend);
#endif
		}
		fn_startBackend = nullptr;
	}

	std::string m_backendName;

	typedef std::vector<iBackend*>* (*startBackend_t)(uint32_t threadOffset, miner_work& pWork, environment& env);

	startBackend_t fn_startBackend = nullptr;

#ifdef WIN32
	HINSTANCE libBackend;
#else
	void* libBackend = nullptr;
#endif
};

} // namespace xmrstak
