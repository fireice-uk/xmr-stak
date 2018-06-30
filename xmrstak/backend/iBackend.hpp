#pragma once

#include "xmrstak/backend/globalStates.hpp"

#include <atomic>
#include <cstdint>
#include <climits>
#include <vector>
#include <string>

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

		iBackend() : iHashCount(0), iTimestamp(0)
		{
		}
	};

} // namespace xmrstak
