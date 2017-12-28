#pragma once

#include <cstddef>

/** pin memory to NUMA node
 *
 * Set the default memory policy for the current thread to bind memory to the
 * NUMA node.
 *
 * @param core_id id of the core to which the memory should be pinned
 */
void bindMemoryToNUMANode( int core_id );
