#pragma once

#include "console.h"
#include <hwloc.h>

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
		printer::inst()->print_str("\n**************** Copy&Paste BEGIN ****************\n\n");
		printer::inst()->print_str("\"cpu_threads_conf\" :\n[\n");

		int depth;
		hwloc_topology_t topology;
		hwloc_obj_t socket;


		hwloc_topology_init(&topology);
		hwloc_topology_load(topology);

		depth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
		if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
		{
			printf("*** The number of sockets is unknown\n");
		}

		for (int i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth); i++)
		{
			socket = hwloc_get_obj_by_depth(topology, depth, i);

			// search cacheprinter::inst()->print_str("\n**************** Copy&Paste ****************\n\n");
			for (int j = 0; j < socket->arity; j++)
			{
				hwloc_obj_t nextLvl = socket->children[j];
				findCache(topology, nextLvl);
			}
		}

		/* Destroy topology object. */
		hwloc_topology_destroy(topology);
		
		printer::inst()->print_str("],\n\n**************** Copy&Paste END ****************\n");
	}

private:

	inline void getConfig(hwloc_topology_t topology, hwloc_obj_t obj, size_t& numHashes, int& numCachesLeft)
	{
		if (obj->type == HWLOC_OBJ_CORE)
		{
			if (obj)
			{
				hwloc_cpuset_t cpuset;
				/* Get a copy of its cpuset that we may modify. */
				cpuset = hwloc_bitmap_dup(obj->cpuset);
				size_t allcpu = hwloc_bitmap_to_ulong(cpuset);
				/* Get only one logical processor (in case the core is
				   SMT/hyperthreaded). */
				hwloc_bitmap_singlify(cpuset);


				int firstNativeCore = hwloc_bitmap_first(cpuset);

				int nativeCores = hwloc_bitmap_weight(cpuset);
				int numPus = obj->arity;
				for (int i = 0; i < numPus && numHashes != 0 && firstNativeCore != -1; i++)
				{
					hwloc_obj_t pu = obj->children[i];
					// only use native pu's
					if (pu->type == HWLOC_OBJ_PU && hwloc_bitmap_isset( cpuset, i + firstNativeCore ))
					{
						// if no cache is available we give each native core a hash
						int numUnit = numCachesLeft != 0 ? numCachesLeft : nativeCores;

						// two hashes per native pu if number of hashes if larger than compute units
						int power = numHashes > numUnit ? 2 : 1;
						char strbuf[256];
	
						snprintf(strbuf, sizeof(strbuf), "   { \"low_power_mode\" : %s, \"no_prefetch\" : true, \"affine_to_cpu\" : %u },\n",
							power == 2 ? "true" : "false", pu->os_index);
						printer::inst()->print_str(strbuf);

						// update number of free hashes
						numHashes -= power;

						// one cache is filled with hashes
						if (numCachesLeft != 0) numCachesLeft--;
					}
				}
			}
		}
		else
		{
			for (int i = 0; i < obj->arity; i++)
				getConfig(topology, obj->children[i], numHashes, numCachesLeft);
		}
	}

	inline void findCache(hwloc_topology_t topology, hwloc_obj_t obj)
	{
		if (obj->type == HWLOC_OBJ_CACHE)
		{
			size_t cacheSize = obj->attr->cache.size;
			size_t numHashL3 = ( cacheSize + m_scratchPadMemSize/ 2llu ) / m_scratchPadMemSize;

			/* check cache is exclusive or inclusive */
			const char* value = hwloc_obj_get_info_by_name(obj, "Inclusive");


			bool doL3 = true;
			if (value == NULL || value[0] != 49 || cacheSize == 0)
			{
				size_t numHashes = 0;
				int numL2 = obj->arity;
				for (int k = 0; k < numL2; k++)
				{
					hwloc_obj_t l3Cache = obj->children[k];
					size_t l2Cache = 0;

					if (obj->type == HWLOC_OBJ_CACHE)
						l2Cache = l3Cache->attr->cache.size;
					else
						break;

					if (l2Cache < m_scratchPadMemSize)
					{
						// we need to start from L3
						break;
					}

					// start from L2

					/* if more hashes available than objects in the current depth of the topology
					 * than divide with round down else round up
					 */
					int extraHash = numHashL3 > numL2 ? numHashL3 / numL2 : (numHashL3 + numL2 - 1) / numL2;
					numHashL3 -= extraHash;
					if (numHashL3 < 0)
						numHashL3 = 0;
					numHashes += extraHash;
					//add L2 hashes
					numHashes += ( l2Cache + m_scratchPadMemSize / 2llu ) / m_scratchPadMemSize;
					int numCachesLeft = numL2;
					getConfig(topology, l3Cache, numHashes, numCachesLeft);
					doL3 = false;
				}
			}
			if (doL3)
			{
				int numCachesLeft = obj->arity;
				getConfig(topology, obj, numHashL3, numCachesLeft);
			}
		}
		else
			for (int j = 0; j < obj->arity; j++)
				findCache(topology, obj->children[j]);
	}

	static constexpr size_t m_scratchPadMemSize = ( 2llu * 1024llu * 1024llu );
};
