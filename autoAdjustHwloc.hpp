#pragma once

#include "console.h"
#include <hwloc.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif // _WIN32

class autoAdjust
{
public:

	autoAdjust()
	{
	}

	void printConfig()
	{
		printer::inst()->print_str("The configuration for 'cpu_threads_conf' in your config file is 'null'.\n");
		printer::inst()->print_str("The miner evaluates your system and prints a suggestion for the section `cpu_threads_conf` to the terminal.\n");
		printer::inst()->print_str("The values are not optimal, please try to tweak the values based on notes in config.txt.\n");
		printer::inst()->print_str("Please copy & paste the block within the asterisks to your config.\n\n");

		hwloc_topology_t topology;
		hwloc_topology_init(&topology);
		hwloc_topology_load(topology);

		try
		{
			std::vector<hwloc_obj_t> tlcs;
			tlcs.reserve(16);
			results.reserve(16);

			findChildrenCaches(hwloc_get_root_obj(topology),
				[&tlcs](hwloc_obj_t found) { tlcs.emplace_back(found); } );

			if(tlcs.size() == 0)
				throw(std::runtime_error("The CPU doesn't seem to have a cache."));

			for(hwloc_obj_t obj : tlcs)
				proccessTopLevelCache(obj);

			printer::inst()->print_str("\n**************** Copy&Paste BEGIN ****************\n\n");
			printer::inst()->print_str("\"cpu_threads_conf\" :\n[\n");

			for(uint32_t id : results)
			{
				char str[128];
				snprintf(str, sizeof(str), "    { \"low_power_mode\" : %s, \"no_prefetch\" : true, \"affine_to_cpu\" : %u },\n",
					(id & 0x8000000) != 0 ? "true" : "false", id & 0x7FFFFFF);
				printer::inst()->print_str(str);
			}

			printer::inst()->print_str("],\n\n**************** Copy&Paste END ****************\n");
		}
		catch(const std::runtime_error& err)
		{
			printer::inst()->print_msg(L0, "Autoconf FAILED: %s", err.what());
			printer::inst()->print_str("\nPrinting config for a single thread. Please try to add new ones until the hashrate slows down.\n");
			printer::inst()->print_str("\n**************** FAILURE Copy&Paste BEGIN ****************\n\n");
			printer::inst()->print_str("\"cpu_threads_conf\" :\n[\n");
			printer::inst()->print_str("    { \"low_power_mode\" : false, \"no_prefetch\" : true, \"affine_to_cpu\" : false },\n");
			printer::inst()->print_str("],\n\n**************** FAILURE Copy&Paste END ****************\n");
		}

		/* Destroy topology object. */
		hwloc_topology_destroy(topology);
	}

private:
	static constexpr size_t hashSize = 2 * 1024 * 1024;
	std::vector<uint32_t> results;

	template<typename func>
	inline void findChildrenByType(hwloc_obj_t obj, hwloc_obj_type_t type, func lambda)
	{
		for(size_t i=0; i < obj->arity; i++)
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

	template<typename func>
	inline void findChildrenCaches(hwloc_obj_t obj, func lambda)
	{
		for(size_t i=0; i < obj->arity; i++)
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
	void proccessTopLevelCache(hwloc_obj_t obj)
	{
		if(obj->attr == nullptr)
			throw(std::runtime_error("Cache object hasn't got attributes."));

		size_t PUs = 0;
		findChildrenByType(obj, HWLOC_OBJ_PU, [&PUs](hwloc_obj_t found) { PUs++; } );

		//Strange case, but we will handle it silently, surely there must be one PU somewhere?
		if(PUs == 0)
			return;

		if(obj->attr->cache.size == 0)
		{
			//We will always have one child if PUs > 0
			if(!isCacheObject(obj->children[0]))
				throw(std::runtime_error("The CPU doesn't seem to have a cache."));

			//Try our luck with lower level caches
			for(size_t i=0; i < obj->arity; i++)
				proccessTopLevelCache(obj->children[i]);
			return;
		}

		size_t cacheSize = obj->attr->cache.size;
		if(isCacheExclusive(obj))
		{
			for(size_t i=0; i < obj->arity; i++)
			{
				hwloc_obj_t l2obj = obj->children[i];
				//If L2 is exclusive and greater or equal to 2MB add room for one more hash
				if(isCacheObject(l2obj) && l2obj->attr != nullptr && l2obj->attr->cache.size >= hashSize)
					cacheSize += hashSize;
			}
		}

		std::vector<hwloc_obj_t> cores;
		cores.reserve(16);
		findChildrenByType(obj, HWLOC_OBJ_CORE, [&cores](hwloc_obj_t found) { cores.emplace_back(found); } );

		size_t cacheHashes = (cacheSize + hashSize/2) / hashSize;

		//Firstly allocate PU 0 of every CORE, then PU 1 etc.
		size_t pu_id = 0;
		while(cacheHashes > 0 && PUs > 0)
		{
			bool allocated_pu = false;
			for(hwloc_obj_t core : cores)
			{
				if(core->arity <= pu_id || core->children[pu_id]->type != HWLOC_OBJ_PU)
					continue;

				size_t os_id = core->children[pu_id]->os_index;

				if(cacheHashes > PUs)
				{
					cacheHashes -= 2;
					os_id |= 0x8000000; //double hash marker bit
				}
				else
					cacheHashes--;
				PUs--;

				allocated_pu = true;
				results.emplace_back(os_id);

				if(cacheHashes == 0)
					break;
			}

			if(!allocated_pu)
				throw(std::runtime_error("Failed to allocate a PU."));

			pu_id++;
		}
	}
};
