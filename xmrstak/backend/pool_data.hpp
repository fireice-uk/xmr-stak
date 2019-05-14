#pragma once

#include <cstdint>
#include <string>

constexpr static size_t invalid_pool_id = (-1);

namespace xmrstak
{

struct pool_data
{
	uint32_t iSavedNonce;
	size_t pool_id;

	pool_data() :
		iSavedNonce(0),
		pool_id(invalid_pool_id)
	{
	}
};

} // namespace xmrstak
