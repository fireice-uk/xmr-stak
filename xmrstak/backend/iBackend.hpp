#pragma once

#include "xmrstak/backend/globalStates.hpp"

#include <atomic>
#include <cstdint>
#include <climits>
#include <vector>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 
#include <windows.h>
#else
#include <pthread.h>
#endif

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
		BackendType backendType = UNKNOWN;
#ifdef _WIN32
		unsigned long mining_thread_id;
#else
		pthread_t mining_thread_id;
#endif // _WIN32

		iBackend() : iHashCount(0), iTimestamp(0)
		{
		}
		
		void wait()
		{
#ifdef _WIN32
			HANDLE hThread = OpenThread(SYNCHRONIZE, FALSE, mining_thread_id);
			if (hThread)
			{
				WaitForSingleObject(hThread, INFINITE);
				CloseHandle(hThread);
			}
#else 
			pthread_join(mining_thread_id, NULL);
#endif // _WIN32

		}
		
		~iBackend()
		{
			wait();
		}
		
	};

} // namespace xmrstak
