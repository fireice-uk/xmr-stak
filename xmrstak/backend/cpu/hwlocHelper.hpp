#pragma once

#include <vector>
#include <cstddef>

#ifndef CONF_NO_HWLOC
#	include <hwloc.h>
#endif

/** pin memory to NUMA node
 *
 * Set the default memory policy for the current thread to bind memory to the
 * NUMA node.
 *
 * @param puId core id
 */
void hwlocBind(size_t puId);

/** get numa node id based on a thread id
 *
 * @return 0 if no numa node is found, else numa id (zero based)
 */
size_t numdaId(size_t puId);

/** number of numa nodes
 *
 * @return if no numa node is found 1 will be returned
 */
size_t getNumNumaNodes();

#ifndef CONF_NO_HWLOC
	std::vector<hwloc_obj_t> getNumaNodes(hwloc_topology_t topology);
#endif