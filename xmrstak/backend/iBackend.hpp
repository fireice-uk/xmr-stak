#pragma once

#include "xmrstak/backend/globalStates.hpp"

#include <atomic>
#include <cstdint>
#include <climits>
#include <vector>
#include <string>

namespace xmrstak
{
	struct iBackend
	{

		enum BackendType : uint32_t { UNKNOWN = 0, CPU = 1u, AMD = 2u, NVIDIA = 3u };
		
		static std::string getName(const BackendType type)
		{
			std::vector<std::string> backendNames = {
				"UNKNOWN",
				"CPU",
				"AMD",
				"NVIDIA"
			};
			return backendNames[static_cast<uint32_t>(type)];
		}

		std::atomic<uint64_t> iHashCount;
		std::atomic<uint64_t> iTimestamp;
		uint32_t iThreadNo;
		BackendType backendType = UNKNOWN;

		iBackend() : iHashCount(0), iTimestamp(0)
		{
		}
	};

} // namepsace xmrstak
