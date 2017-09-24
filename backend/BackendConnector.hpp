#pragma once
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include "IBackend.hpp"
#include "miner_work.h"

namespace xmrstak
{

	struct BackendConnector
	{
		static std::vector<IBackend*>* thread_starter(miner_work& pWork);
		static bool self_test();
	};

} // namepsace xmrstak
