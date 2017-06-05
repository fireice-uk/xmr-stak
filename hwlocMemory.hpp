#pragma once

#include "console.h"

#ifndef CONF_NO_HWLOC

#include <hwloc.h>

/** pin memory to NUMA node
 *
 * Set the default memory policy for the current thread to bind memory to the
 * NUMA node.
 *
 * @param puId core id
 */
void bindMemoryToNUMANode( int puId )
{
	int depth;
	hwloc_topology_t topology;
	hwloc_obj_t obj;

	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);

	depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

	for( int i = 0;
		i < hwloc_get_nbobjs_by_depth(topology, depth);
		i++ )
	{
		hwloc_obj_t pu = hwloc_get_obj_by_depth(topology, depth, i);
		if(  pu->os_index == puId )
			if( 0 > hwloc_set_membind_nodeset(
				topology,
				pu->nodeset,
				HWLOC_MEMBIND_BIND,
				HWLOC_MEMBIND_THREAD)
			)
				printer::inst()->print_msg(L0, "hwloc: can't bind memory");
			else
			{
				printer::inst()->print_msg(L0, "hwloc: memory pinned");
				break;
			}
	}
}
#else

void bindMemoryToNUMANode( int )
{
}

#endif
