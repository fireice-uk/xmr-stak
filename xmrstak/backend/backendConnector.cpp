/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
  *
  */

#include "iBackend.hpp"
#include "backendConnector.hpp"
#include "miner_work.hpp"
#include "globalStates.hpp"
#include "plugin.hpp"
#include "xmrstak/misc/environment.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"

#include "cpu/minethd.hpp"
#ifndef CONF_NO_CUDA
#	include "nvidia/minethd.hpp"
#endif
#ifndef CONF_NO_OPENCL
#	include "amd/minethd.hpp"
#endif

#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <thread>
#include <bitset>


namespace xmrstak
{

bool BackendConnector::self_test()
{
	return cpu::minethd::self_test();
}

std::vector<iBackend*>* BackendConnector::thread_starter(miner_work& pWork)
{
	globalStates::inst().iGlobalJobNo = 0;
	globalStates::inst().iConsumeCnt = 0;


	std::vector<iBackend*>* pvThreads = new std::vector<iBackend*>;

#ifndef CONF_NO_CUDA
	if(params::inst().useNVIDIA)
	{
		plugin nvidiaplugin("NVIDIA", "xmrstak_cuda_backend");
		std::vector<iBackend*>* nvidiaThreads = nvidiaplugin.startBackend(static_cast<uint32_t>(pvThreads->size()), pWork, environment::inst());
		pvThreads->insert(std::end(*pvThreads), std::begin(*nvidiaThreads), std::end(*nvidiaThreads));
		if(nvidiaThreads->size() == 0)
			printer::inst()->print_msg(L0, "WARNING: backend NVIDIA disabled.");
	}
#endif

#ifndef CONF_NO_OPENCL
	if(params::inst().useAMD)
	{
		plugin amdplugin("AMD", "xmrstak_opencl_backend");
		std::vector<iBackend*>* amdThreads = amdplugin.startBackend(static_cast<uint32_t>(pvThreads->size()), pWork, environment::inst());
		pvThreads->insert(std::end(*pvThreads), std::begin(*amdThreads), std::end(*amdThreads));
		if(amdThreads->size() == 0)
			printer::inst()->print_msg(L0, "WARNING: backend AMD disabled.");
	}
#endif

#ifndef CONF_NO_CPU
	if(params::inst().useCPU)
	{
		auto cpuThreads = cpu::minethd::thread_starter(static_cast<uint32_t>(pvThreads->size()), pWork);
		pvThreads->insert(std::end(*pvThreads), std::begin(cpuThreads), std::end(cpuThreads));
		if(cpuThreads.size() == 0)
			printer::inst()->print_msg(L0, "WARNING: backend CPU disabled.");
	}
#endif

	globalStates::inst().iThreadCount = pvThreads->size();
	return pvThreads;
}

} // namespace xmrstak
