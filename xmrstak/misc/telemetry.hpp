#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>

namespace xmrstak
{

class telemetry
{
public:
	telemetry(size_t iThd);
	void push_perf_value(size_t iThd, uint64_t iHashCount, uint64_t iTimestamp);
	double calc_telemetry_data(size_t iLastMillisec, size_t iThread);

private:
	std::mutex* mtx;
	constexpr static size_t iBucketSize = 2 << 11; //Power of 2 to simplify calculations
	constexpr static size_t iBucketMask = iBucketSize - 1;
	uint32_t* iBucketTop;
	uint64_t** ppHashCounts;
	uint64_t** ppTimestamps;
};

} // namespace xmrstak
