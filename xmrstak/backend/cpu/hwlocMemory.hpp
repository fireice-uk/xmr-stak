#pragma once

#include <cstddef>

/** pin memory to NUMA node
 *
 * Set the default memory policy for the current thread to bind memory to the
 * NUMA node.
 *
 * @param puId core id
 */
void bindMemoryToNUMANode(size_t puId);
