
#include "cryptonight.h"
#include "randomx/randomx.h"


void* getRandomXDataset(const size_t numaId)
{
	return randomx_get_dataset_memory(randomX_global_ctx::inst().getDataset(numaId));
}


uint64_t getRandomXDatasetSize()
{
	return randomx_dataset_item_count() * 64;
}
