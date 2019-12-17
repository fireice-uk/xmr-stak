#pragma once

#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/backend/cpu/hwlocHelper.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif // _WIN32

#include <string>

#include <hwloc.h>
#include <stdio.h>
#include <algorithm>

namespace xmrstak
{
namespace cpu
{

class autoAdjustHwloc
{
public:
	autoAdjustHwloc()
	{
		auto neededAlgorithms = ::jconf::inst()->GetCurrentCoinSelection().GetAllAlgorithms();

		for(const auto algo : neededAlgorithms)
		{
			l3MemRequire = std::max(l3MemRequire, algo.L3());
			l2MemRequire = std::max(l2MemRequire, algo.L2());
		}
	}

	bool printConfig()
	{

		hwloc_topology_t topology;
		hwloc_topology_init(&topology);
		if(hwloc_topology_load(topology) < 0)
			return false;

		std::string conf;
		configEditor configTpl{};

		// load the template of the backend config into a char variable
		const char* tpl =
#include "./config.tpl"
			;
		configTpl.set(std::string(tpl));

		bool is_successful = true;
		try
		{

			std::vector<hwloc_obj_t> tlcs;
			findChildrenCaches(hwloc_get_root_obj(topology),
				[&tlcs](hwloc_obj_t found) { tlcs.emplace_back(found); });

			if(tlcs.size() == 0)
				throw(std::runtime_error("The CPU doesn't seem to have a cache."));
			printer::inst()->print_msg(LDEBUG,"process %u cache elements", uint32_t(tlcs.size()));
			for(hwloc_obj_t obj : tlcs)
				processTopLevelCache(obj);


			for(const auto& thd : threads)
			{
				conf += std::string("    { \"low_power_mode\" : ");
				conf += std::to_string(thd.num_hashes);
				conf += std::string(", \"affine_to_cpu\" : ");
				conf += std::to_string(thd.core_id);
				conf += std::string(" },\n");
			}
		}
		catch(const std::runtime_error& err)
		{
			is_successful = false;
			printer::inst()->print_msg(L0, "Autoconf with hwloc FAILED: %s. Trying basic autoconf.", err.what());
		}

		configTpl.replace("CPUCONFIG", conf);
		configTpl.write(params::inst().configFileCPU);
		printer::inst()->print_msg(L0, "CPU configuration stored in file '%s'", params::inst().configFileCPU.c_str());
		/* Destroy topology object. */
		hwloc_topology_destroy(topology);

		return is_successful;
	}

  private:
	size_t l3MemRequire = 0;
	size_t l2MemRequire = 0;

	struct Thread
	{
		Thread(const uint32_t c_id, const uint32_t n_hash) :
			core_id(c_id), num_hashes(n_hash)
		{}

		uint32_t core_id = 0;
		uint32_t num_hashes = 1;
	};

	std::vector<Thread> threads;

	template <typename func>
	inline void findChildrenByType(hwloc_obj_t obj, hwloc_obj_type_t type, func lambda)
	{
		for(size_t i = 0; i < obj->arity; i++)
		{
			if(obj->children[i]->type == type)
				lambda(obj->children[i]);
			else
				findChildrenByType(obj->children[i], type, lambda);
		}
	}

	inline bool isCacheObject(hwloc_obj_t obj)
	{
#if HWLOC_API_VERSION >= 0x20000
		return hwloc_obj_type_is_cache(obj->type);
#else
		return obj->type == HWLOC_OBJ_CACHE;
#endif // HWLOC_API_VERSION
	}

	template <typename func>
	inline void findChildrenCaches(hwloc_obj_t obj, func lambda)
	{
		for(size_t i = 0; i < obj->arity; i++)
		{
			if(isCacheObject(obj->children[i]))
				lambda(obj->children[i]);
			else
				findChildrenCaches(obj->children[i], lambda);
		}
	}

	inline bool isCacheExclusive(hwloc_obj_t obj)
	{
		const char* value = hwloc_obj_get_info_by_name(obj, "Inclusive");
		return value == nullptr || value[0] != '1';
	}

	// Top level cache isn't shared with other cores on the same package
	// This will usually be 1 x L3, but can be 2 x L2 per package
	void processTopLevelCache(hwloc_obj_t obj)
	{
		if(obj->attr == nullptr)
			throw(std::runtime_error("Cache object hasn't got attributes."));

		size_t numPUs = 0;
		findChildrenByType(obj, HWLOC_OBJ_PU, [&numPUs](hwloc_obj_t found) { numPUs++; });

		//Strange case, but we will handle it silently, surely there must be one PU somewhere?
		if(numPUs == 0)
			return;

		if(obj->attr->cache.size == 0)
		{
			//We will always have one child if numPUs > 0
			if(!isCacheObject(obj->children[0]))
				throw(std::runtime_error("The CPU doesn't seem to have a cache."));

			//Try our luck with lower level caches
			for(size_t i = 0; i < obj->arity; i++)
				processTopLevelCache(obj->children[i]);
			return;
		}

		size_t l3CacheSize = obj->attr->cache.size;
		size_t numL2Caches = obj->arity;
		bool isExclusive = isCacheExclusive(obj);
		size_t l2CacheSize = 0u;
		if(obj->attr->cache.depth == 3)
		{
			for(size_t i = 0; i < numL2Caches; i++)
			{
				hwloc_obj_t l2obj = obj->children[i];
				if(isCacheObject(l2obj) && l2obj->attr)
				{
					//If L3 is exclusive and greater or equal to 2MB add room for one more hash
					if(isExclusive && l2obj->attr->cache.size >= l3MemRequire)
						l3CacheSize += l3MemRequire;
					else
						l2CacheSize += l2obj->attr->cache.size;
				}
			}
		}

		size_t l2CacheSizePerHash = l2CacheSize / numL2Caches;
		printer::inst()->print_msg(LDEBUG,"%u L3 cache, required per hash %u", uint32_t(l3CacheSize), uint32_t(l3MemRequire));
		printer::inst()->print_msg(LDEBUG,"%u L2 cache, required per hash %u", uint32_t(l2CacheSize), uint32_t(l2MemRequire));

		size_t l3CacheHashes = std::max(l3CacheSize / l3MemRequire, size_t(1u));
		size_t l2CacheHashes = std::max(l2CacheSizePerHash / l2MemRequire, size_t(1u)) * numL2Caches;

		// we have no lvl2 cache or our top lvl cache is L2
		if(l2CacheSize == 0u)
			l2CacheHashes = l3CacheHashes;

		std::vector<hwloc_obj_t> cores;
		cores.reserve(16);
		findChildrenByType(obj, HWLOC_OBJ_CORE, [&cores](hwloc_obj_t found) { cores.emplace_back(found); });

		printer::inst()->print_msg(LDEBUG,"%u L3 hash limit", uint32_t(l3CacheHashes));
		printer::inst()->print_msg(LDEBUG,"%u L2 hash limit", uint32_t(l2CacheHashes));
		printer::inst()->print_msg(LDEBUG,"%u PU(s) available", uint32_t(numPUs));
		size_t numHashCacheLimited = std::min(l2CacheHashes, l3CacheHashes);
		// do not use more PUs than available
		size_t usePus = std::min(numHashCacheLimited, numPUs);

		// currently do not use multi hash per PU (all tests has shown it is slower)
		//size_t numHashesPerPu = std::max(numHashCacheLimited / numPUs, size_t(1u));
		size_t numHashesPerPu = 1u;

		printer::inst()->print_msg(LDEBUG,"use %u PU(s)", uint32_t(usePus));
		printer::inst()->print_msg(LDEBUG,"use %u hashe(s) per pu", uint32_t(numHashesPerPu));

		//Firstly allocate PU 0 of every CORE, then PU 1 etc.
		size_t pu_id = 0;
		while(usePus > 0)
		{
			bool allocated_pu = false;
			for(hwloc_obj_t core : cores)
			{
				if(core->arity <= pu_id || core->children[pu_id]->type != HWLOC_OBJ_PU)
					continue;

				size_t os_id = core->children[pu_id]->os_index;

				allocated_pu = true;
				threads.emplace_back(Thread(os_id, numHashesPerPu));

				usePus--;

				if(usePus == 0)
					break;
			}

			if(!allocated_pu)
				throw(std::runtime_error("Failed to allocate a PU."));

			pu_id++;
		}
	}
};

} // namespace cpu
} // namespace xmrstak
