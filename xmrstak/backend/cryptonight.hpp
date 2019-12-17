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
	randomX_arqma = 4

	//cryptonight_turtle = start_derived_algo_id,
	// please add the algorithm name to get_algo_name()
};

/** get name of the algorithm
 *
 * @param algo mining algorithm
 */
inline std::string get_algo_name(xmrstak_algo_id algo_id)
{
	static std::array<std::string, 5> base_algo_names =
		{{
			"invalid_algo",
			"randomx",
			"randomx_loki",
			"randomx_wow",
			"randomx_arqma"
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

	xmrstak_algo(xmrstak_algo_id name_id, xmrstak_algo_id algorithm, size_t l3, size_t l2, size_t l1) :
		algo_name(name_id),
		base_algo(algorithm),
		m_l3(l3),
		m_l2(l2),
		m_l1(l1)
	{
	}

	/** check if the algorithm is equal to another algorithm
	 *
	 * we do not check the member algo_name because this is only an alias name
	 */
	bool operator==(const xmrstak_algo& other) const
	{
		return other.Id() == Id() && other.L3() == L3() && other.L2() == L2() && other.L1() == L1();
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

	size_t L3() const
	{
		return m_l3;
	}

	size_t L2() const
	{
		return m_l2;
	}

	size_t L1() const
	{
		return m_l1;
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



	xmrstak_algo_id algo_name = invalid_algo;
	xmrstak_algo_id base_algo = invalid_algo;
	size_t m_l3 = 1u; // avoid diffision by zero
	size_t m_l2 = 1u;
	size_t m_l1 = 1u;
};

// default cryptonight
constexpr size_t _2MiB = 2 * 1024 * 1024;
constexpr size_t _256KiB = 256 * 1024;
constexpr size_t _16KiB = 16 * 1024;

constexpr uint32_t RX_ARQMA_ITER = 0x10000;

inline xmrstak_algo POW(xmrstak_algo_id algo_id)
{
	static std::array<xmrstak_algo, 5> pow = {{
		{invalid_algo},
		{randomX, randomX, _2MiB, _256KiB, _16KiB},
		{randomX_loki, randomX_loki, _2MiB, _256KiB, _16KiB},
		{randomX_wow, randomX_wow, _2MiB/2, _256KiB/2, _16KiB},
		{randomX_arqma, randomX_arqma, _2MiB/8, _256KiB/2, _16KiB}
	}};

	return pow[algo_id];
}
