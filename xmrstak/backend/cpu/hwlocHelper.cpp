#include "xmrstak/misc/console.hpp"
#include "hwlocHelper.hpp"

#ifndef CONF_NO_HWLOC

#include <hwloc.h>

inline int
xmrstak_set_membind_nodeset(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
#if HWLOC_API_VERSION >= 0x20000
	return hwloc_set_membind(
		topology,
		nodeset,
		policy,
		flags| HWLOC_MEMBIND_BYNODESET);
#else
	return hwloc_set_membind_nodeset(
		topology,
		nodeset,
		policy,
		flags);
#endif
}

hwloc_obj_t getPU(hwloc_topology_t topology, size_t puId)
{
	hwloc_obj_t result = nullptr;
	int pu_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
	uint32_t chunks = hwloc_get_nbobjs_by_depth(topology, pu_depth);

	for(uint32_t i = 0; i < chunks;	i++)
	{
		hwloc_obj_t pu = hwloc_get_obj_by_depth(topology, pu_depth, i);
		if(pu->os_index == puId)
		{
			result = pu;
			break;
		}
	}

	return result;
}

/** pin memory to NUMA node and thread to given core
 *
 * Set the default memory policy for the current thread to bind memory to the
 * NUMA node.
 *
 * @param puId core id
 */
void hwlocBind(size_t puId)
{
	int depth;
	hwloc_topology_t topology;

	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);

	hwloc_bitmap_t puBitMap = hwloc_bitmap_alloc();
	hwloc_bitmap_set(puBitMap, puId);
	if(0 > hwloc_set_cpubind(topology, puBitMap, HWLOC_CPUBIND_THREAD))
		printer::inst()->print_msg(L0, "hwloc: pu bind to %u failed", uint32_t(puId));
	hwloc_bitmap_free(puBitMap);

	if(!hwloc_topology_get_support(topology)->membind->set_thisthread_membind)
	{
		printer::inst()->print_msg(L0, "hwloc: set_thisthread_membind not supported");
		hwloc_topology_destroy(topology);
		return;
	}

	depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

	hwloc_obj_t puPtr = getPU(topology, puId);
	if(puPtr != nullptr)
	{
		if(0 > xmrstak_set_membind_nodeset(
				   topology,
				   puPtr->nodeset,
				   HWLOC_MEMBIND_BIND,
				   HWLOC_MEMBIND_THREAD))
		{
			printer::inst()->print_msg(L0, "hwloc: can't bind memory");
		}
		else
		{
			printer::inst()->print_msg(L0, "hwloc: memory pinned");
		}
	}

	hwloc_topology_destroy(topology);
}


size_t numdaId(size_t puId)
{
	size_t result = 0;

	hwloc_topology_t topology;
	hwloc_topology_init(&topology);
	if(hwloc_topology_load(topology) < 0)
		return result;

	hwloc_obj_t puPtr = getPU(topology, puId);
	if(puPtr != nullptr)
	{
		int numa_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
		uint32_t chunks = hwloc_get_nbobjs_by_depth(topology, numa_depth);
		for(uint32_t i = 0; i < chunks; i++)
		{
			hwloc_obj_t numa = hwloc_get_obj_by_depth(topology, numa_depth, i);
			if(hwloc_bitmap_isset(puPtr->nodeset, numa->os_index))
			{
				result = i;
				printer::inst()->print_msg(LDEBUG,"PU %u is on numa %u", uint32_t(puId), i);
				break;
			}
		}
	}
	else
	{
		printer::inst()->print_msg(LDEBUG,"PU %u not found", uint32_t(puId));
	}

	hwloc_topology_destroy(topology);
	return result;
}

std::vector<hwloc_obj_t> getNumaNodes(hwloc_topology_t topology)
{
	int numa_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
	uint32_t chunks = hwloc_get_nbobjs_by_depth(topology, numa_depth);
	printer::inst()->print_msg(LDEBUG,"%u numa node(s) found", chunks);
	std::vector<hwloc_obj_t> result(chunks);

	for(uint32_t i = 0; i < chunks; i++)
		result[i] = hwloc_get_obj_by_depth(topology, numa_depth, i);

	return result;
}

size_t getNumNumaNodes()
{
	size_t result = 1;

	hwloc_topology_t topology;
	hwloc_topology_init(&topology);
	if(hwloc_topology_load(topology) < 0)
		return result;

	std::vector<hwloc_obj_t> num_nodes = getNumaNodes(topology);
	result = num_nodes.size();
	if(result == 0)
		result = 1;

	hwloc_topology_destroy(topology);
	return result;
}

#else

void hwlocBind(size_t)
{
}

size_t getNumNumaNodes()
{
	return 1;
}

size_t numdaId(size_t puId)
{
	return 0;
}

#endif
