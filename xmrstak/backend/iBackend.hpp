#pragma once

#include "xmrstak/backend/globalStates.hpp"
#include "xmrstak/net/msgstruct.hpp"

#include <atomic>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

template <typename T, std::size_t N>
constexpr std::size_t countof(T const (&)[N]) noexcept
{
	return N;
}

namespace xmrstak
{
struct iBackend
{

	enum BackendType : uint32_t
	{
		UNKNOWN = 0u,
		CPU = 1u,
		AMD = 2u,
		NVIDIA = 3u
	};

	static const char* getName(const BackendType type)
	{
		const char* backendNames[] = {
			"unknown",
			"cpu",
			"amd",
			"nvidia"};

		uint32_t i = static_cast<uint32_t>(type);
		if(i >= countof(backendNames))
			i = 0;

		return backendNames[i];
	}

	std::atomic<uint64_t> iHashCount;
	std::atomic<uint64_t> iTimestamp;
	uint32_t iThreadNo;
	uint32_t iGpuIndex;
	BackendType backendType = UNKNOWN;
	uint64_t iLastStamp = get_timestamp_ms();
	double avgHashPerMsec = 0.0;

	void updateStats(uint64_t numNewHashes, size_t poolId)
	{
		uint64_t iStamp = get_timestamp_ms();
		double timeDiff = static_cast<double>(iStamp - iLastStamp);
		iLastStamp = iStamp;

		if(poolId == 0)
		{
			// if dev pool is active interpolate the number of shares (avoid hash rate drops)
			numNewHashes = static_cast<uint64_t>(avgHashPerMsec * timeDiff);
		}
		else
		{
			const double hashRatePerMs = static_cast<double>(numNewHashes) / timeDiff;
			constexpr double averagingBias = 0.1;
			avgHashPerMsec = avgHashPerMsec * (1.0 - averagingBias) + hashRatePerMs * averagingBias;
		}
		iHashCount.fetch_add(numNewHashes, std::memory_order_relaxed);
		iTimestamp.store(iStamp, std::memory_order_relaxed);
	}

	iBackend() :
		iHashCount(0),
		iTimestamp(0)
	{
	}
};

} // namespace xmrstak
