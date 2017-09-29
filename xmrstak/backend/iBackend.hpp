#pragma once

#include <atomic>
#include <cstdint>


namespace xmrstak
{

	struct IBackend
	{
		std::atomic<uint64_t> iHashCount;
		std::atomic<uint64_t> iTimestamp;

		IBackend() : iHashCount(0), iTimestamp(0)
		{
		}
	};

} // namepsace xmrstak
