#pragma once

#include "xmrstak/misc/environment.hpp"
#include "xmrstak/params.hpp"

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include "iBackend.hpp"
#include <iostream>

#ifndef USE_PRECOMPILED_HEADERS
#	ifdef WIN32
#		include <direct.h>
#		include <windows.h>
#	else
#		include <sys/types.h>
#		include <dlfcn.h>
#	endif
#	include <iostream>
#endif

namespace xmrstak
{

struct plugin
{

	plugin(const std::string backendName, const std::string libName) : m_backendName(backendName)
	{
#ifdef WIN32
		libBackend = LoadLibrary(TEXT((libName + ".dll").c_str()));
		if(!libBackend)
		{
			std::cerr << "WARNING: "<< m_backendName <<" cannot load backend library: " << (libName + ".dll") << std::endl;
			return;
		}
#else
		// `.so` linux file extention for dynamic libraries
		std::string fileExtension = ".so";
#	if defined(__APPLE__)
		// `.dylib` Mac OS X file extention for dynamic libraries
		fileExtension = ".dylib";
#	endif
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
			std::cerr << "WARNING: "<< m_backendName <<" cannot load backend library: " << dlerror() << std::endl;
			return;
		}
#endif

#ifdef WIN32
		fn_starterBackend = (starterBackend_t) GetProcAddress(libBackend, "xmrstak_start_backend");
		if (!fn_starterBackend)
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_start_backend': " <<GetLastError()<< std::endl;
		fn_versionBackend = (versionBackend_t) GetProcAddress(libBackend, "xmrstak_version_backend");
		if (!fn_versionBackend)
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_version_backend': " <<GetLastError()<< std::endl;
#else
		// reset last error
		dlerror();
		fn_starterBackend = (starterBackend_t) dlsym(libBackend, "xmrstak_start_backend");
		const char* dlsym_error = dlerror();
		if(dlsym_error)
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_start_backend': " << dlsym_error << std::endl;
		dlerror();
		fn_versionBackend = (versionBackend_t) dlsym(libBackend, "xmrstak_version_backend");
		dlsym_error = dlerror();
		if(dlsym_error)
			std::cerr << "WARNING: backend plugin " << libName << " contains no entry 'xmrstak_version_backend': " << dlsym_error << std::endl;
		
#endif
	}

	std::vector<iBackend*>* startBackend(uint32_t threadOffset, miner_work& pWork, environment& env)
	{
		if(fn_starterBackend == nullptr)
		{
			std::vector<iBackend*>* pvThreads = new std::vector<iBackend*>();
			std::cerr << "WARNING: " << m_backendName << " Backend disabled"<< std::endl;
			return pvThreads;
		}

		return fn_starterBackend(threadOffset, pWork, env);
	}

	std::string getVersion()
	{
		if(fn_starterBackend == nullptr)
		{
			printer::inst()->print_msg(L1, "WARNING: extension %s has no version number, please update all miner files.", m_backendName.c_str());
			return std::string("unknown plugin version");
		}

		return fn_versionBackend();
	}

	std::string m_backendName;

	typedef std::vector<iBackend*>* (*starterBackend_t)(uint32_t threadOffset, miner_work& pWork, environment& env);
	typedef std::string (*versionBackend_t)();

	starterBackend_t fn_starterBackend = nullptr;
	versionBackend_t fn_versionBackend = nullptr;

#ifdef WIN32
	HINSTANCE libBackend;
#else
	void *libBackend;
#endif

/* \todo add unload to destructor and change usage of plugin that libs keeped open until the miner endss
#ifdef WIN32
	FreeLibrary(libBackend);
#else
	dlclose(libBackend);
#endif
 * */
};

} // namespace xmrstak
