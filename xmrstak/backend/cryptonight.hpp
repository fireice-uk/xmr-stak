#pragma once
#include <stddef.h>
#include <inttypes.h>
#include <type_traits>

enum xmrstak_algo
{
	invalid_algo = 0,
	cryptonight = 1,
	cryptonight_lite = 2,
	cryptonight_monero = 3,
	cryptonight_heavy = 4,
	cryptonight_aeon = 5,
	cryptonight_ipbc = 6, // equal to cryptonight_aeon with a small tweak in the miner code
	cryptonight_stellite = 7, //equal to cryptonight_monero but with one tiny change
	cryptonight_masari = 8, //equal to cryptonight_monero but with less iterations, used by masari
	cryptonight_haven = 9, // equal to cryptonight_heavy with a small tweak
	cryptonight_bittube2 = 10, // derived from cryptonight_heavy with own aes-round implementation and minor other tweaks
	cryptonight_monero_v8 = 11,
	cryptonight_superfast = 12,
	cryptonight_gpu = 13,
	cryptonight_turtle = 14,
	cryptonight_zelerius = 15 // equal to cryptonight_monero_v8 but with less iterations
};

// define aeon settings
constexpr size_t CRYPTONIGHT_LITE_MEMORY = 1 * 1024 * 1024;
constexpr uint32_t CRYPTONIGHT_LITE_MASK = 0xFFFF0;
constexpr uint32_t CRYPTONIGHT_LITE_ITER = 0x40000;

constexpr size_t CRYPTONIGHT_MEMORY = 2 * 1024 * 1024;
constexpr uint32_t CRYPTONIGHT_MASK = 0x1FFFF0;
constexpr uint32_t CRYPTONIGHT_ITER = 0x80000;

constexpr size_t CRYPTONIGHT_HEAVY_MEMORY = 4 * 1024 * 1024;
constexpr uint32_t CRYPTONIGHT_HEAVY_MASK = 0x3FFFF0;
constexpr uint32_t CRYPTONIGHT_HEAVY_ITER = 0x40000;

constexpr uint32_t CRYPTONIGHT_GPU_MASK = 0x1FFFC0;
constexpr uint32_t CRYPTONIGHT_GPU_ITER = 0xC000;

constexpr uint32_t CRYPTONIGHT_MASARI_ITER = 0x40000;

constexpr uint32_t CRYPTONIGHT_SUPERFAST_ITER = 0x20000;

constexpr size_t CRYPTONIGHT_TURTLE_MEMORY = 256 * 1024;
constexpr uint32_t CRYPTONIGHT_TURTLE_MASK = 0x1FFF0;
constexpr uint32_t CRYPTONIGHT_TURTLE_ITER = 0x10000;

constexpr size_t CRYPTONIGHT_ZELERIUS_MEMORY = 2 * 1024 * 1024;
constexpr uint32_t CRYPTONIGHT_ZELERIUS_MASK = 0x1FFFF0;
constexpr uint32_t CRYPTONIGHT_ZELERIUS_ITER = 0x60000;

template<xmrstak_algo ALGO>
inline constexpr size_t cn_select_memory() { return 0; }

template<>
inline constexpr size_t cn_select_memory<cryptonight>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_lite>() { return CRYPTONIGHT_LITE_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_monero>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_monero_v8>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_heavy>() { return CRYPTONIGHT_HEAVY_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_aeon>() { return CRYPTONIGHT_LITE_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_ipbc>() { return CRYPTONIGHT_LITE_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_stellite>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_masari>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_haven>() { return CRYPTONIGHT_HEAVY_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_bittube2>() { return CRYPTONIGHT_HEAVY_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_superfast>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_gpu>() { return CRYPTONIGHT_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_turtle>() { return CRYPTONIGHT_TURTLE_MEMORY; }

template<>
inline constexpr size_t cn_select_memory<cryptonight_zelerius>() { return CRYPTONIGHT_ZELERIUS_MEMORY; }

inline size_t cn_select_memory(xmrstak_algo algo)
{
	switch(algo)
	{
	case cryptonight_stellite:
	case cryptonight_monero:
	case cryptonight_monero_v8:
	case cryptonight_masari:
	case cryptonight:
	case cryptonight_superfast:
	case cryptonight_gpu:
		return CRYPTONIGHT_MEMORY;
	case cryptonight_ipbc:
	case cryptonight_aeon:
	case cryptonight_lite:
		return CRYPTONIGHT_LITE_MEMORY;
	case cryptonight_bittube2:
	case cryptonight_haven:
	case cryptonight_heavy:
		return CRYPTONIGHT_HEAVY_MEMORY;
	case cryptonight_turtle:
		return CRYPTONIGHT_TURTLE_MEMORY;
	case cryptonight_zelerius:
		return CRYPTONIGHT_ZELERIUS_MEMORY;
	default:
		return 0;
	}
}

template<xmrstak_algo ALGO>
inline constexpr uint32_t cn_select_mask() { return 0; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight>() { return CRYPTONIGHT_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_lite>() { return CRYPTONIGHT_LITE_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_monero>() { return CRYPTONIGHT_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_monero_v8>() { return CRYPTONIGHT_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_heavy>() { return CRYPTONIGHT_HEAVY_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_aeon>() { return CRYPTONIGHT_LITE_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_ipbc>() { return CRYPTONIGHT_LITE_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_stellite>() { return CRYPTONIGHT_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_masari>() { return CRYPTONIGHT_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_haven>() { return CRYPTONIGHT_HEAVY_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_bittube2>() { return CRYPTONIGHT_HEAVY_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_superfast>() { return CRYPTONIGHT_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_gpu>() { return CRYPTONIGHT_GPU_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_turtle>() { return CRYPTONIGHT_TURTLE_MASK; }

template<>
inline constexpr uint32_t cn_select_mask<cryptonight_zelerius>() { return CRYPTONIGHT_ZELERIUS_MASK; }

inline size_t cn_select_mask(xmrstak_algo algo)
{
	switch(algo)
	{
	case cryptonight_stellite:
	case cryptonight_monero:
	case cryptonight_monero_v8:
	case cryptonight_masari:
	case cryptonight:
	case cryptonight_superfast:
		return CRYPTONIGHT_MASK;
	case cryptonight_ipbc:
	case cryptonight_aeon:
	case cryptonight_lite:
		return CRYPTONIGHT_LITE_MASK;
	case cryptonight_bittube2:
	case cryptonight_haven:
	case cryptonight_heavy:
		return CRYPTONIGHT_HEAVY_MASK;
	case cryptonight_gpu:
		return CRYPTONIGHT_GPU_MASK;
	case cryptonight_turtle:
		return CRYPTONIGHT_TURTLE_MASK;
	case cryptonight_zelerius:
		return CRYPTONIGHT_ZELERIUS_MASK;
	default:
		return 0;
	}
}

template<xmrstak_algo ALGO>
inline constexpr uint32_t cn_select_iter() { return 0; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight>() { return CRYPTONIGHT_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_lite>() { return CRYPTONIGHT_LITE_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_monero>() { return CRYPTONIGHT_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_monero_v8>() { return CRYPTONIGHT_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_heavy>() { return CRYPTONIGHT_HEAVY_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_aeon>() { return CRYPTONIGHT_LITE_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_ipbc>() { return CRYPTONIGHT_LITE_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_stellite>() { return CRYPTONIGHT_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_masari>() { return CRYPTONIGHT_MASARI_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_haven>() { return CRYPTONIGHT_HEAVY_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_bittube2>() { return CRYPTONIGHT_HEAVY_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_superfast>() { return CRYPTONIGHT_SUPERFAST_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_gpu>() { return CRYPTONIGHT_GPU_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_turtle>() { return CRYPTONIGHT_TURTLE_ITER; }

template<>
inline constexpr uint32_t cn_select_iter<cryptonight_zelerius>() { return CRYPTONIGHT_ZELERIUS_ITER; }

inline size_t cn_select_iter(xmrstak_algo algo)
{
	switch(algo)
	{
	case cryptonight_stellite:
	case cryptonight_monero:
	case cryptonight_monero_v8:
	case cryptonight:
		return CRYPTONIGHT_ITER;
	case cryptonight_ipbc:
	case cryptonight_aeon:
	case cryptonight_lite:
		return CRYPTONIGHT_LITE_ITER;
	case cryptonight_bittube2:
	case cryptonight_haven:
	case cryptonight_heavy:
		return CRYPTONIGHT_HEAVY_ITER;
	case cryptonight_masari:
		return CRYPTONIGHT_MASARI_ITER;
	case cryptonight_superfast:
		return CRYPTONIGHT_SUPERFAST_ITER;
	case cryptonight_gpu:
		return CRYPTONIGHT_GPU_ITER;
	case cryptonight_turtle:
		return CRYPTONIGHT_TURTLE_ITER;
	case cryptonight_zelerius:
		return CRYPTONIGHT_ZELERIUS_ITER;
	default:
		return 0;
	}
}
