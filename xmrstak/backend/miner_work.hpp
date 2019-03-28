#pragma once

#include "xmrstak/backend/pool_data.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

namespace xmrstak
{
struct miner_work
{
	char sJobID[64];
	uint8_t bWorkBlob[128];
	uint32_t iWorkSize;
	uint64_t iTarget;
	bool bNiceHash;
	bool bStall;
	size_t iPoolId;
	uint64_t iBlockHeight;
	uint8_t* ref_ptr;

	miner_work() :
		iWorkSize(0),
		bNiceHash(false),
		bStall(true),
		iPoolId(invalid_pool_id),
		ref_ptr((uint8_t*)&iBlockHeight) {}

	miner_work(const char* sJobID, const uint8_t* bWork, uint32_t iWorkSize,
		uint64_t iTarget, bool bNiceHash, size_t iPoolId, uint64_t iBlockHeiht) :
		iWorkSize(iWorkSize),
		iTarget(iTarget),
		bNiceHash(bNiceHash),
		bStall(false),
		iPoolId(iPoolId),
		iBlockHeight(iBlockHeiht),
		ref_ptr((uint8_t*)&iBlockHeight)
	{
		assert(iWorkSize <= sizeof(bWorkBlob));
		memcpy(this->bWorkBlob, bWork, iWorkSize);
		memcpy(this->sJobID, sJobID, sizeof(miner_work::sJobID));
	}

	miner_work(miner_work&& from) :
		iWorkSize(from.iWorkSize),
		iTarget(from.iTarget),
		bStall(from.bStall),
		iPoolId(from.iPoolId),
		iBlockHeight(from.iBlockHeight),
		ref_ptr((uint8_t*)&iBlockHeight)
	{
		assert(iWorkSize <= sizeof(bWorkBlob));
		memcpy(bWorkBlob, from.bWorkBlob, iWorkSize);
		memcpy(this->sJobID, sJobID, sizeof(miner_work::sJobID));
	}

	miner_work(miner_work const&) = delete;

	miner_work& operator=(miner_work&& from)
	{
		assert(this != &from);

		iBlockHeight = from.iBlockHeight;
		iPoolId = from.iPoolId;
		bStall = from.bStall;
		iWorkSize = from.iWorkSize;
		bNiceHash = from.bNiceHash;
		iTarget = from.iTarget;

		assert(iWorkSize <= sizeof(bWorkBlob));
		memcpy(sJobID, from.sJobID, sizeof(sJobID));
		memcpy(bWorkBlob, from.bWorkBlob, iWorkSize);

		return *this;
	}

	miner_work& operator=(miner_work const& from)
	{
		assert(this != &from);

		iBlockHeight = from.iBlockHeight;
		iPoolId = from.iPoolId;
		bStall = from.bStall;
		iWorkSize = from.iWorkSize;
		bNiceHash = from.bNiceHash;
		iTarget = from.iTarget;

		if(!ref_ptr)
			return *this;

		for(size_t i = 0; i <= 7 && iPoolId; i++)
			ref_ptr[i] = from.ref_ptr[7 - i];

		assert(iWorkSize <= sizeof(bWorkBlob));
		memcpy(sJobID, from.sJobID, sizeof(sJobID));
		memcpy(bWorkBlob, from.bWorkBlob, iWorkSize);

		return *this;
	}

	uint8_t getVersion() const
	{
		return bWorkBlob[0];
	}
};
} // namespace xmrstak
