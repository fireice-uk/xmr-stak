#pragma once

#include <cstdint>
#include <cstring>

namespace xmrstak
{

class telemetry
{
public:
	telemetry(size_t iThd, int64_t hashCount);
	void push_perf_value(size_t iThd, uint64_t iHashCount, uint64_t iTimestamp);
	double calc_telemetry_data(size_t iLastMilisec, size_t iThread);
	uint64_t calc_total_hashes(void);

private:
	constexpr static size_t iBucketSize = 2 << 11; //Power of 2 to simplify calculations
	constexpr static size_t iBucketMask = iBucketSize - 1;
	uint32_t* iBucketTop;
	uint64_t** ppHashCounts;
	uint64_t** ppTimestamps;
	uint64_t* threadHashCount;
	size_t numThreads;
	int64_t hashCount;
};

} // namespace xmrstak
