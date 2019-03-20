#pragma once

#include "xmrstak/backend/globalStates.hpp"

#include <atomic>
#include <cstdint>
#include <climits>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <mutex>

template <typename T, std::size_t N>
constexpr std::size_t countof(T const (&)[N]) noexcept
{
	return N;
}

namespace xmrstak
{
	struct iBackend
	{

		enum BackendType : uint32_t { UNKNOWN = 0u, CPU = 1u, AMD = 2u, NVIDIA = 3u };

		static const char* getName(const BackendType type)
		{
			const char* backendNames[] = {
				"unknown",
				"cpu",
				"amd",
				"nvidia"
			};

			uint32_t i = static_cast<uint32_t>(type);
			if(i >= countof(backendNames))
				i = 0;

			return backendNames[i];
		}

		std::atomic<uint64_t> iHashCount;
		std::atomic<uint64_t> iTimestamp;
		uint32_t iThreadNo;
		uint64_t thdNo;
		BackendType backendType = UNKNOWN;
		std::thread oWorkThd;
		std::promise<void> numa_promise;
		std::promise<void> thread_work_promise;
		// block thread until all NVIDIA GPUs are initialized
		std::future<void> thread_work_guard;
		std::promise<void> order_fix;
		std::mutex thd_aff_set;
		// volatile becaues the compiler may optimize it in minethd when it sees that is isn't changed there
		volatile std::atomic<bool> bSuspend;
		volatile std::atomic<bool> bQuit;
		volatile std::atomic<bool> pause_idle;

		iBackend() : iHashCount(0), iTimestamp(0), bSuspend(false), bQuit(false), pause_idle(false)
		{
		}

		void join()
		{
			if (oWorkThd.joinable())
				oWorkThd.join();
		}

		~iBackend()
		{
			bQuit = true;
			join();
		}

	};

} // namespace xmrstak
