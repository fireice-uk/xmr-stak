#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include "IBackend.hpp"
#include <iostream>

#ifndef USE_PRECOMPILED_HEADERS
#ifdef WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <dlfcn.h>
#endif
#include <iostream>
#endif

namespace xmrstak
{

struct Plugin
{

	Plugin(const std::string backendName, const std::string libName) : fn_starterBackend(nullptr), m_backendName(backendName)
	{
#ifdef WIN32
		libBackend = LoadLibrary(TEXT((libName + ".dll").c_str()));
		if(!libBackend)
		{
			std::cerr << "WARNING: "<< m_backendName <<" cannot load backend library: " << (libName + ".dll") << std::endl;
			return;
		}
#else
		libBackend = dlopen((std::string("lib") + libName + ".so").c_str(), RTLD_LAZY);
		if(!libBackend)
		{
			std::cerr << "WARNING: "<< m_backendName <<" cannot load backend library: " << dlerror() << std::endl;
			return;
		}
#endif

#ifdef WIN32
		fn_starterBackend = (starterBackend_t) GetProcAddress(libBackend, "xmrstak_start_backend");
		if (!fn_starterBackend)
		{
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_start_backend': " <<GetLastError()<< std::endl;
		}
#else
		// reset last error
		dlerror();
		fn_starterBackend = (starterBackend_t) dlsym(libBackend, "xmrstak_start_backend");
		const char* dlsym_error = dlerror();
		if(dlsym_error)
		{
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_start_backend': " << dlsym_error << std::endl;
		}
#endif
	}

	std::vector<IBackend*>* startBackend(uint32_t threadOffset, miner_work& pWork)
	{
		if(fn_starterBackend == nullptr)
		{
			std::vector<IBackend*>* pvThreads = new std::vector<IBackend*>();
			std::cerr << "WARNING: " << m_backendName << " Backend disabled"<< std::endl;
			return pvThreads;
		}

		return fn_starterBackend(threadOffset, pWork);
	}

	std::string m_backendName;

	typedef std::vector<IBackend*>* (*starterBackend_t)(uint32_t threadOffset, miner_work& pWork);

	starterBackend_t fn_starterBackend;

#ifdef WIN32
    HINSTANCE libBackend;
#else
    void *libBackend;
#endif

/* \todo add unload to destructor and change usage of Plugin that libs keeped open until the miner endss
#ifdef WIN32
    FreeLibrary(libBackend);
#else
    dlclose(libBackend);
#endif
 * */
};

} // namepsace xmrstak
