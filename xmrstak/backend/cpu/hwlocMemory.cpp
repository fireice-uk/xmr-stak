#include "xmrstak/backend/cpu/hwlocMemory.hpp"

#ifndef CONF_NO_HWLOC

#include "xmrstak/misc/console.hpp"

#include <hwloc.h>

/** pin memory to NUMA node
 *
 * Set the default memory policy for the current thread to bind memory to the
 * NUMA node.
 *
 * @param core_id id of the core to which the memory should be pinned
 */
void bindMemoryToNUMANode( int core_id )
{

	if(core_id < 0)
	{
		printer::inst()->print_msg(L0, "hwloc: WARNING bindMemoryToNUMANode called with negative core id");
		return;
	}
	uint32_t puId = static_cast<uint32_t>(core_id);
	hwloc_topology_t topology;

	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);

	if(!hwloc_topology_get_support(topology)->membind->set_thisthread_membind)
	{
		printer::inst()->print_msg(L0, "hwloc: set_thisthread_membind not supported");
		hwloc_topology_destroy(topology);
		return;
	}

	uint32_t depth = 0u;
	int returnedDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

	// do not interpret error codes a size
	if(returnedDepth >=0 )
		depth = static_cast<uint32_t>(returnedDepth);

	for( uint32_t i = 0u;
		i < hwloc_get_nbobjs_by_depth(topology, depth);
		i++ )
	{
		hwloc_obj_t pu = hwloc_get_obj_by_depth(topology, depth, i);
		if(  pu->os_index == puId )
		{
			if( 0 > hwloc_set_membind_nodeset(
				topology,
				pu->nodeset,
				HWLOC_MEMBIND_BIND,
				HWLOC_MEMBIND_THREAD))
			{
				printer::inst()->print_msg(L0, "hwloc: can't bind memory");
			}
			else
			{
				printer::inst()->print_msg(L0, "hwloc: memory pinned");
				break;
			}
		}
	}

	hwloc_topology_destroy(topology);
}
#else

void bindMemoryToNUMANode( int )
{
}

#endif
