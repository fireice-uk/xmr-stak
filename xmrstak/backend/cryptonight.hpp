#pragma once
#include <array>
#include <inttypes.h>
#include <stddef.h>
#include <string>
#include <type_traits>

constexpr size_t start_derived_algo_id = 1000;

enum xmrstak_algo_id
{
	invalid_algo = 0,
	randomX = 1,
	randomX_loki = 2,
	randomX_wow = 3,

	//cryptonight_turtle = start_derived_algo_id,
	// please add the algorithm name to get_algo_name()
};

/** get name of the algorithm
 *
 * @param algo mining algorithm
 */
inline std::string get_algo_name(xmrstak_algo_id algo_id)
{
	static std::array<std::string, 4> base_algo_names =
		{{
			"invalid_algo",
			"randomx",
			"randomx_loki",
			"randomx_wow"
		}};

	static std::array<std::string, 0> derived_algo_names =
		{{	}};

	if(algo_id < start_derived_algo_id)
		return base_algo_names[algo_id];
	else
		return derived_algo_names[algo_id - start_derived_algo_id];
}

struct xmrstak_algo
{
	xmrstak_algo(xmrstak_algo_id name_id) :
		algo_name(name_id),
		base_algo(name_id)
	{
	}
	xmrstak_algo(xmrstak_algo_id name_id, xmrstak_algo_id algorithm) :
		algo_name(name_id),
		base_algo(algorithm)
	{
	}
	xmrstak_algo(xmrstak_algo_id name_id, xmrstak_algo_id algorithm, uint32_t iteration) :
		algo_name(name_id),
		base_algo(algorithm),
		iter(iteration)
	{
	}
	xmrstak_algo(xmrstak_algo_id name_id, xmrstak_algo_id algorithm, uint32_t iteration, size_t memory) :
		algo_name(name_id),
		base_algo(algorithm),
		iter(iteration),
		mem(memory)
	{
	}
	xmrstak_algo(xmrstak_algo_id name_id, xmrstak_algo_id algorithm, uint32_t iteration, size_t memory, uint32_t mem_mask) :
		algo_name(name_id),
		base_algo(algorithm),
		iter(iteration),
		mem(memory),
		mask(mem_mask)
	{
	}

	/** check if the algorithm is equal to another algorithm
	 *
	 * we do not check the member algo_name because this is only an alias name
	 */
	bool operator==(const xmrstak_algo& other) const
	{
		return other.Id() == Id() && other.Mem() == Mem() && other.Iter() == Iter() && other.Mask() == Mask();
	}

	bool operator==(const xmrstak_algo_id& id) const
	{
		return base_algo == id;
	}

	operator xmrstak_algo_id() const
	{
		return base_algo;
	}

	xmrstak_algo_id Id() const
	{
		return base_algo;
	}

	xmrstak_algo_id Algo_Id() const
	{
		return algo_name;
	}

	size_t Mem() const
	{
		if(base_algo == invalid_algo)
			return 0;
		else
			return mem;
	}

	uint32_t Iter() const
	{
		return iter;
	}

	/** Name of the algorithm
	 *
	 * This name is only an alias for the native implemented base algorithm.
	 */
	std::string Name() const
	{
		return get_algo_name(algo_name);
	}

	/** Name of the parent algorithm
	 *
	 * This is the real algorithm which is implemented in all POW functions.
	 */
	std::string BaseName() const
	{
		return get_algo_name(base_algo);
	}

	uint32_t Mask() const
	{
		// default is a 16 byte aligne mask
		if(mask == 0)
			return ((mem - 1u) / 16) * 16;
		else
			return mask;
	}

	xmrstak_algo_id algo_name = invalid_algo;
	xmrstak_algo_id base_algo = invalid_algo;
	uint32_t iter = 0u;
	size_t mem = 0u;
	uint32_t mask = 0u;
};

// default cryptonight
constexpr size_t CN_MEMORY = 2 * 1024 * 1024;
constexpr uint32_t CN_ITER = 0x80000;
constexpr uint32_t CN_MASK = ((CN_MEMORY - 1) / 16) * 16;

// crptonight gpu
constexpr uint32_t CN_GPU_MASK = 0x1FFFC0;
constexpr uint32_t CN_GPU_ITER = 0xC000;

// cryptonight turtle (the mask is not using the full 256kib scratchpad)
constexpr uint32_t CN_TURTLE_MASK = 0x1FFF0;

constexpr uint32_t CN_ZELERIUS_ITER = 0x60000;

constexpr uint32_t CN_WALTZ_ITER = 0x60000;

constexpr uint32_t CN_DOUBLE_ITER = 0x100000;

inline xmrstak_algo POW(xmrstak_algo_id algo_id)
{
	static std::array<xmrstak_algo, 4> pow = {{{invalid_algo, invalid_algo},
		{randomX, randomX, CN_ITER, CN_MEMORY},
		{randomX_loki, randomX_loki, CN_ITER, CN_MEMORY},
		{randomX_wow, randomX_wow, CN_ITER, CN_MEMORY/2}
	}};

	return pow[algo_id];
}
