# Tuning Guide

## Content Overview
* [NVIDIA Backend](#nvidia-backend)
  * [Choose Value for `threads` and `blocks`](#choose-value-for-threads-and-blocks)
  * [Add more GPUs](#add-more-gpus)
* [AMD Backend](#amd-backend)
  * [Choose `intensity` and `worksize`](#choose-intensity-and-worksize)
  * [Add more GPUs](#add-more-gpus)
  * [Increase Memory Pool](#increase-memory-pool)

## NVIDIA Backend

By default the NVIDIA backend can be tuned in the config file `nvidia.txt`

### Choose Value for `threads` and `blocks`

The optimal parameter for the `threads` and `blocks` option in `config.txt` depend on your GPU.
For all GPU's with a compute capability `>=2.0` and `<6.0` there is a restriction of the amount of RAM that can be used for the mining algorithm.
The maximum RAM that can be used must be less than 2GB (e.g. GTX TITAN) or 1GB (e.g. GTX 750-TI).
The amount of RAM used for mining can be changed with `"threads" : T, "blocks : B"`.
  - `T` = threads used per block
  - `B` = CUDA blocks started (should be a multiple of the multiprocessors `M` on the GPU)

For the 2GB limit the equations must be full filled: `T * B * 2 <= 1900` and ` B mod M == 0`.
The value `1900` is used because there is a little data overhead for administration.
The GTX Titan X has 24 multiprocessors `M`, this means a valid and good starting configuration is `"threads" : 16, "blocks : 48"`
and full fill all restrictions `16 * 48 * 2 = 1536` and `48 mod 24 = 0`.

The memory limit for NVIDIA Pascal GPUs is `16` GiB if the newest CUDA driver is used.

### Add More GPUs

To add a new GPU you need to add a new config set to `gpu_threads_conf`.
`index` is the number of the gpu, the index order not follow the order from `nvidia-smi` or the order shown in windows.

```
"gpu_threads_conf" :
[
    { "index" : 0, "threads" : 17, "blocks" : 60, "bfactor" : 0, "bsleep" :  0, "affine_to_cpu" : false},
    { "index" : 1, "threads" : 17, "blocks" : 60, "bfactor" : 0, "bsleep" :  0, "affine_to_cpu" : false},
],
```

## AMD Backend

By default the AMD backend can be tuned in the config file `amd.txt`

### Choose `intensity` and `worksize`

Intensity means the number of threads used to mine. The maximum intensity is GPU_MEMORY_MB / 2 - 128, however for cards with 4GB and more, the optimum is likely to be lower than that.
`worksize` is the number of threads working together to increase the miner performance.
In the most cases a `worksize` of `16` or `8` is optimal.

### Add More GPUs

To add a new GPU you need to add a new config set to `gpu_threads_conf`. `index` is the OpenCL index of the gpu.
`platform_index`is the index of the OpenCL platform (Intel / AMD / Nvidia).
If you are unsure of either GPU or platform index value, you can use `clinfo` tool that comes with AMD APP SDK to dump the values.

```
"gpu_threads_conf" :
[
    { "index" : 0, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false },
    { "index" : 1, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false },
],

"platform_index" : 0,
```

### Increase Memory Pool

By setting the following environment variables before the miner is started OpenCl allows the miner to more threads.
This variables must be set each time before the miner is started else it could be that the miner can not allocate enough memory and is crashing.

```
export GPU_FORCE_64BIT_PTR=1
export GPU_MAX_HEAP_SIZE=99
export GPU_MAX_ALLOC_PERCENT=99
export GPU_SINGLE_ALLOC_PERCENT=99
```

*Note:* Windows user must use `set` instead of `export` to define an environment variable.