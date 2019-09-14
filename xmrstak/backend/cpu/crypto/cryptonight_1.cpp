
#include "cryptonight.h"
#include "randomx/randomx.h"


void* getRandomXDataset()
{
	return randomx_get_dataset_memory(randomX_global_ctx::inst().getDataset());
}


uint64_t getRandomXDatasetSize()
{
	return randomx_dataset_item_count() * 64;
}
