#pragma once

#include "xmrstak/backend/globalStates.hpp"

#include <atomic>
#include <cstdint>
#include <climits>


namespace xmrstak
{
	// only allowed for unsigned value \todo add static assert
	template<typename T>
	T reverseBits(T value)
	{
		/* init with value (to get LSB) */
		T result = value;
		/* extra shift needed at end */
		int s = sizeof(T) * CHAR_BIT - 1;
		for (value >>= 1; value; value >>= 1)
		{
			result <<= 1;
			result |= value & 1;
			s--;
		}
		/* shift when values highest bits are zero */
		result <<= s;
		return result;
	}

	struct iBackend
	{
		inline uint32_t calc_start_nonce(uint32_t resume)
		{
			return reverseBits<uint32_t>(static_cast<uint32_t>(iThreadNo + globalStates::inst().iThreadCount * resume));
		}

		// Limited version of the nonce calc above
		inline uint32_t calc_nicehash_nonce(uint32_t start, uint32_t resume)
		{
			return start | ( calc_start_nonce(resume) >> 8u );
		}

		std::atomic<uint64_t> iHashCount;
		std::atomic<uint64_t> iTimestamp;
		uint32_t iThreadNo;

		iBackend() : iHashCount(0), iTimestamp(0)
		{
		}
	};

} // namepsace xmrstak
