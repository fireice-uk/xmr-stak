#pragma once

#include "iBackend.hpp"
#include "miner_work.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace xmrstak
{

struct BackendConnector
{
	static std::vector<iBackend*>* thread_starter(miner_work& pWork);
	static bool self_test();
};

} // namespace xmrstak
