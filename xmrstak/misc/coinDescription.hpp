#pragma once

#include "xmrstak/backend/cryptonight.hpp"

#include <stdlib.h>
#include <string>


namespace xmrstak
{
	struct coinDescription
	{
		xmrstak_algo algo = xmrstak_algo::invalid_algo;
		xmrstak_algo algo_root = xmrstak_algo::invalid_algo;
		uint8_t fork_version = 0u;

		coinDescription() = default;

		coinDescription(const xmrstak_algo in_algo, xmrstak_algo in_algo_root, const uint8_t in_fork_version) :
			algo(in_algo), algo_root(in_algo_root), fork_version(in_fork_version)
		{}

		inline xmrstak_algo GetMiningAlgo() const { return algo; }
		inline xmrstak_algo GetMiningAlgoRoot() const { return algo_root; }
		inline uint8_t GetMiningForkVersion() const { return fork_version; }
	};

	struct coin_selection
	{
		const char* coin_name = nullptr;
		/* [0] -> user pool
		 * [1] -> dev pool
		 */
		coinDescription pool_coin[2];
		const char* default_pool = nullptr;

		coin_selection() = default;

		coin_selection(
			const char* in_coin_name,
			const coinDescription user_coinDescription,
			const coinDescription dev_coinDescription,
			const char* in_default_pool
		) :
			coin_name(in_coin_name), default_pool(in_default_pool)
		{
			pool_coin[0] = user_coinDescription;
			pool_coin[1] = dev_coinDescription;
		}

		/** get coin description for the pool
		 *
		 * @param poolId 0 select dev pool, else the user pool is selected
		 */
		inline coinDescription GetDescription(size_t poolId) const {
			coinDescription tmp = (poolId == 0 ? pool_coin[1] : pool_coin[0]);
			return tmp;
		}
	};
} // namespace xmrstak
