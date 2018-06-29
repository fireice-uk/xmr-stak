#pragma once

#include "xmrstak/misc/console.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/backend/cryptonight.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif // _WIN32

#include <string>

#include <hwloc.h>
#include <stdio.h>


namespace xmrstak
{
namespace cpu
{

class autoAdjust
{
public:

	autoAdjust()
	{
		hashMemSize = std::max(
			cn_select_memory(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgo()),
			cn_select_memory(::jconf::inst()->GetCurrentCoinSelection().GetDescription(1).GetMiningAlgoRoot())
		);
		halfHashMemSize = hashMemSize / 2u;
	}

	bool printConfig()
	{

		hwloc_topology_t topology;
		hwloc_topology_init(&topology);
		hwloc_topology_load(topology);

		std::string conf;
		configEditor configTpl{};

		// load the template of the backend config into a char variable
		const char *tpl =
			#include "./config.tpl"
		;
		configTpl.set( std::string(tpl) );

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
				processTopLevelCache(obj);

			for(uint32_t id : results)
			{
				conf += std::string("    { \"low_power_mode\" : ");
				conf += std::string((id & 0x8000000) != 0 ? "true" : "false");
				conf += std::string(", \"no_prefetch\" : true, \"affine_to_cpu\" : ");
				conf += std::to_string(id & 0x7FFFFFF);
				conf += std::string(" },\n");
			}
		}
		catch(const std::runtime_error& err)
		{
			// \todo add fallback to default auto adjust
			conf += std::string("    { \"low_power_mode\" : false, \"no_prefetch\" : true, \"affine_to_cpu\" : false },\n");
			printer::inst()->print_msg(L0, "Autoconf FAILED: %s. Create config for a single thread.", err.what());
		}

		configTpl.replace("CPUCONFIG",conf);
		configTpl.write(params::inst().configFileCPU);
		printer::inst()->print_msg(L0, "CPU configuration stored in file '%s'", params::inst().configFileCPU.c_str());
		/* Destroy topology object. */
		hwloc_topology_destroy(topology);

		return true;
	}

private:
	size_t hashMemSize;
	size_t halfHashMemSize;

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
	void processTopLevelCache(hwloc_obj_t obj)
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
				processTopLevelCache(obj->children[i]);
			return;
		}

		size_t cacheSize = obj->attr->cache.size;
		if(isCacheExclusive(obj))
		{
			for(size_t i=0; i < obj->arity; i++)
			{
				hwloc_obj_t l2obj = obj->children[i];
				//If L2 is exclusive and greater or equal to 2MB add room for one more hash
				if(isCacheObject(l2obj) && l2obj->attr != nullptr && l2obj->attr->cache.size >= hashMemSize)
					cacheSize += hashMemSize;
			}
		}

		std::vector<hwloc_obj_t> cores;
		cores.reserve(16);
		findChildrenByType(obj, HWLOC_OBJ_CORE, [&cores](hwloc_obj_t found) { cores.emplace_back(found); } );

		size_t cacheHashes = (cacheSize + halfHashMemSize) / hashMemSize;

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

} // namespace cpu
} // namespace xmrstak
