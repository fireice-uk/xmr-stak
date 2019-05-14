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
	cryptonight = 1,
	cryptonight_lite = 2,
	cryptonight_monero = 3,
	cryptonight_heavy = 4,
	cryptonight_aeon = 5,
	cryptonight_ipbc = 6,	  // equal to cryptonight_aeon with a small tweak in the miner code
	cryptonight_stellite = 7,  //equal to cryptonight_monero but with one tiny change
	cryptonight_masari = 8,	//equal to cryptonight_monero but with less iterations, used by masari
	cryptonight_haven = 9,	 // equal to cryptonight_heavy with a small tweak
	cryptonight_bittube2 = 10, // derived from cryptonight_heavy with own aes-round implementation and minor other tweaks
	cryptonight_monero_v8 = 11,
	cryptonight_superfast = 12,
	cryptonight_gpu = 13,
	cryptonight_conceal = 14,
	cryptonight_r_wow = 15,
	cryptonight_r = 16,
	cryptonight_v8_reversewaltz = 17, //equal to cryptonight_monero_v8 but with 3/4 iterations and reversed shuffle operation

	cryptonight_turtle = start_derived_algo_id,
	cryptonight_v8_half = (start_derived_algo_id + 1),
	cryptonight_v8_zelerius = (start_derived_algo_id + 2),
	cryptonight_v8_double = (start_derived_algo_id + 3)
	// please add the algorithm name to get_algo_name()
};

/** get name of the algorithm
 *
 * @param algo mining algorithm
 */
inline std::string get_algo_name(xmrstak_algo_id algo_id)
{
	static std::array<std::string, 18> base_algo_names =
		{{
			"invalid_algo",
			"cryptonight",
			"cryptonight_lite",
			"cryptonight_v7",
			"cryptonight_heavy",
			"cryptonight_lite_v7",
			"cryptonight_lite_v7_xor",
			"cryptonight_v7_stellite",
			"cryptonight_masari",
			"cryptonight_haven",
			"cryptonight_bittube2",
			"cryptonight_v8",
			"cryptonight_superfast",
			"cryptonight_gpu",
			"cryptonight_conceal",
			"cryptonight_r_wow",
			"cryptonight_r",
			"cryptonight_v8_reversewaltz" // used by graft
		}};

	static std::array<std::string, 4> derived_algo_names =
		{{"cryptonight_turtle",
			"cryptonight_v8_half", // used by masari and stellite
			"cryptonight_v8_zelerius",
			"cryptonight_v8_double"}};

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
	static std::array<xmrstak_algo, 18> pow = {{{invalid_algo, invalid_algo},
		{cryptonight, cryptonight, CN_ITER, CN_MEMORY},
		{cryptonight_lite, cryptonight_lite, CN_ITER / 2, CN_MEMORY / 2},
		{cryptonight_monero, cryptonight_monero, CN_ITER, CN_MEMORY},
		{cryptonight_heavy, cryptonight_heavy, CN_ITER / 2, CN_MEMORY * 2},
		{cryptonight_aeon, cryptonight_aeon, CN_ITER / 2, CN_MEMORY / 2},
		{cryptonight_ipbc, cryptonight_ipbc, CN_ITER / 2, CN_MEMORY / 2},		  // equal to cryptonight_aeon with a small tweak in the miner code
		{cryptonight_stellite, cryptonight_stellite, CN_ITER, CN_MEMORY},		  //equal to cryptonight_monero but with one tiny change
		{cryptonight_masari, cryptonight_masari, CN_ITER / 2, CN_MEMORY},		  //equal to cryptonight_monero but with less iterations, used by masari
		{cryptonight_haven, cryptonight_haven, CN_ITER / 2, CN_MEMORY * 2},		  // equal to cryptonight_heavy with a small tweak
		{cryptonight_bittube2, cryptonight_bittube2, CN_ITER / 2, CN_MEMORY * 2}, // derived from cryptonight_heavy with own aes-round implementation and minor other tweaks
		{cryptonight_monero_v8, cryptonight_monero_v8, CN_ITER, CN_MEMORY},
		{cryptonight_superfast, cryptonight_superfast, CN_ITER / 4, CN_MEMORY},
		{cryptonight_gpu, cryptonight_gpu, CN_GPU_ITER, CN_MEMORY, CN_GPU_MASK},
		{cryptonight_conceal, cryptonight_conceal, CN_ITER / 2, CN_MEMORY},
		{cryptonight_r_wow, cryptonight_r_wow, CN_ITER, CN_MEMORY},
		{cryptonight_r, cryptonight_r, CN_ITER, CN_MEMORY},
		{cryptonight_v8_reversewaltz, cryptonight_v8_reversewaltz, CN_WALTZ_ITER, CN_MEMORY}}};

	static std::array<xmrstak_algo, 4> derived_pow =
		{{
			{cryptonight_turtle, cryptonight_monero_v8, CN_ITER / 8, CN_MEMORY / 8, CN_TURTLE_MASK},
			{cryptonight_v8_half, cryptonight_monero_v8, CN_ITER / 2, CN_MEMORY},
			{cryptonight_v8_zelerius, cryptonight_monero_v8, CN_ZELERIUS_ITER, CN_MEMORY},
			{cryptonight_v8_double, cryptonight_monero_v8, CN_DOUBLE_ITER, CN_MEMORY}
			// {cryptonight_derived}
		}};

	if(algo_id < start_derived_algo_id)
		return pow[algo_id];
	else
		return derived_pow[algo_id - start_derived_algo_id];
}
