#pragma once
#include <thread>
#include <atomic>

class telemetry
{
public:
	telemetry(size_t iThd);
	void push_perf_value(size_t iThd, uint64_t iHashCount, uint64_t iTimestamp);
	double calc_telemetry_data(size_t iLastMilisec, size_t iThread);

private:
	constexpr static size_t iBucketSize = 2 << 11; //Power of 2 to simplify calculations
	constexpr static size_t iBucketMask = iBucketSize - 1;
	uint32_t* iBucketTop;
	uint64_t** ppHashCounts;
	uint64_t** ppTimestamps;
};

class minethd
{
public:
	struct miner_work
	{
		char        sJobID[64];
		uint8_t     bWorkBlob[112];
		uint32_t    iWorkSize;
		uint32_t    iResumeCnt;
		uint64_t    iTarget;
		bool        bStall;
		size_t      iPoolId;

		miner_work() : iWorkSize(0), bStall(true), iPoolId(0) { }

		miner_work(const char* sJobID, const uint8_t* bWork, uint32_t iWorkSize, uint32_t iResumeCnt,
			uint64_t iTarget, size_t iPoolId) : iWorkSize(iWorkSize), iResumeCnt(iResumeCnt),
			iTarget(iTarget), bStall(false), iPoolId(iPoolId)
		{
			assert(iWorkSize <= sizeof(bWorkBlob));
			memcpy(this->sJobID, sJobID, sizeof(miner_work::sJobID));
			memcpy(this->bWorkBlob, bWork, iWorkSize);
		}

		miner_work(miner_work const&) = delete;

		miner_work& operator=(miner_work const& from)
		{
			assert(this != &from);

			iWorkSize = from.iWorkSize;
			iResumeCnt = from.iResumeCnt;
			iTarget = from.iTarget;
			bStall = from.bStall;
			iPoolId = from.iPoolId;

			assert(iWorkSize <= sizeof(bWorkBlob));
			memcpy(sJobID, from.sJobID, sizeof(sJobID));
			memcpy(bWorkBlob, from.bWorkBlob, iWorkSize);

			return *this;
		}

		miner_work(miner_work&& from) : iWorkSize(from.iWorkSize), iTarget(from.iTarget),
			bStall(from.bStall), iPoolId(from.iPoolId)
		{
			assert(iWorkSize <= sizeof(bWorkBlob));
			memcpy(sJobID, from.sJobID, sizeof(sJobID));
			memcpy(bWorkBlob, from.bWorkBlob, iWorkSize);
		}

		miner_work& operator=(miner_work&& from)
		{
			assert(this != &from);

			iWorkSize = from.iWorkSize;
			iResumeCnt = from.iResumeCnt;
			iTarget = from.iTarget;
			bStall = from.bStall;
			iPoolId = from.iPoolId;

			assert(iWorkSize <= sizeof(bWorkBlob));
			memcpy(sJobID, from.sJobID, sizeof(sJobID));
			memcpy(bWorkBlob, from.bWorkBlob, iWorkSize);

			return *this;
		}
	};

	static void switch_work(miner_work& pWork);
	static std::vector<minethd*>* thread_starter(miner_work& pWork);
	static bool self_test();

	std::atomic<uint64_t> iHashCount;
	std::atomic<uint64_t> iTimestamp;

private:
	minethd(miner_work& pWork, size_t iNo, bool double_work);

	// We use the top 10 bits of the nonce for thread and resume
	// This allows us to resume up to 128 threads 4 times before
	// we get nonce collisions
	// Bottom 22 bits allow for an hour of work at 1000 H/s
	inline uint32_t calc_start_nonce(uint32_t resume)
		{ return (resume * iThreadCount + iThreadNo) << 22; }

	void work_main();
	void double_work_main();
	void consume_work();

	static std::atomic<uint64_t> iGlobalJobNo;
	static std::atomic<uint64_t> iConsumeCnt;
	static uint64_t iThreadCount;
	uint64_t iJobNo;

	static miner_work oGlobalWork;
	miner_work oWork;

	std::thread oWorkThd;
	uint8_t iThreadNo;

	bool bQuit;
};

