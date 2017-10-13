#pragma once

#include "xmrstak/backend/globalStates.hpp"

#include <atomic>
#include <cstdint>
#include <climits>


namespace xmrstak
{
	struct iBackend
	{
		std::atomic<uint64_t> iHashCount;
		std::atomic<uint64_t> iTimestamp;
		uint32_t iThreadNo;

		iBackend() : iHashCount(0), iTimestamp(0)
		{
		}
	};

} // namepsace xmrstak
