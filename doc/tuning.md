# Tuning Guide

## Content Overview
* [Benchmark](#benchmark)
* [Windows](#windows)
* [NVIDIA Backend](#nvidia-backend)
  * [Choose Value for `threads` and `blocks`](#choose-value-for-threads-and-blocks)
  * [Add more GPUs](#add-more-gpus)
* [AMD Backend](#amd-backend)
  * [Choose `intensity` and `worksize`](#choose-intensity-and-worksize)
  * [Add more GPUs](#add-more-gpus)
  * [Two Threads per GPU](two-threads-per-gpu)
  * [Interleave Tuning](interleave-tuning )
  * [disable comp_mode](#disable-comp_mode)
  * [change the scratchpad memory pattern](change-the-scratchpad-memory-pattern)
  * [Increase Memory Pool](#increase-memory-pool)
  * [Scratchpad Indexing](#scratchpad-indexing)
* [CPU Backend](#cpu-backend)
  * [Choose Value for `low_power_mode`](#choose-value-for-low_power_mode)

## Benchmark
To benchmark the miner speed there are two ways.
  - Mine against a pool end press the key `h` after 30 sec to see the hash report.
  - Start the miner with the cli option `--benchmark BLOCKVERSION`. The miner will not connect to any pool and performs a 60sec performance benchmark with all enabled back-ends.

## Windows
"Run As Administrator" prompt (UAC) confirmation is needed to use large pages on Windows 7.
On Windows 10 it is only needed once to set up the account to use them.
Disable the dialog with the command line option `--noUAC`

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
    { "index" : 0, "threads" : 17, "blocks" : 60, "bfactor" : 0, "bsleep" :  0,
      "affine_to_cpu" : false, "sync_mode" : 3, "mem_mode" : 1,
    },
    { "index" : 1, "threads" : 17, "blocks" : 60, "bfactor" : 0, "bsleep" :  0,
      "affine_to_cpu" : false, "sync_mode" : 3, "mem_mode" : 1,
    },
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
    { "index" : 0, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false,
      "strided_index" : true, "mem_chunk" : 2, "unroll" : 8, "comp_mode" : true,
      "interleave" : 40
    },
    { "index" : 1, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false,
      "strided_index" : true, "mem_chunk" : 2, "unroll" : 8, "comp_mode" : true,
      "interleave" : 40
    },
],

"platform_index" : 0,
```

### Two Threads per GPU

Some GPUs like AMD Vega can mine faster if two threads are using the same GPU.
Use the auto generated config as base and repeat the config entry for a GPU.
If the attribute `index` is used twice than two threads will use one GPU.
Take care that the required memory usage on the GPU will also double.
Therefore adjust your intensity by hand.

```
"gpu_threads_conf" :
[
    { "index" : 0, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false,
      "strided_index" : true, "mem_chunk" : 2, "unroll" : 8, "comp_mode" : true,
      "interleave" : 40
    },
    { "index" : 0, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false,
      "strided_index" : true, "mem_chunk" : 2, "unroll" : 8, "comp_mode" : true,
      "interleave" : 40
    },
],

"platform_index" : 0,
```

### Interleave Tuning

Interleave controls when a worker thread is starting to calculate a bunch of hashes 
if two worker threads are used to utilize one GPU.
This option has no effect if only one worker thread is used per GPU.

![Interleave](img/interleave.png) 

Interleave defines how long a thread needs to wait to start the next hash calculation relative to the last started worker thread.
To choose a interleave value larger than 50% makes no sense because than the gpu will not be utilized well enough.
In the most cases the default 40 is a good value but on some systems e.g. Linux Rocm 1.9.1 driver with RX5XX you need to adjust the value.
If you get many interleave message in a row (over 1 minute) you should adjust the value.

```
OpenCL Interleave 0|1: 642/2400.50 ms - 30.1
OpenCL Interleave 0|0: 355/2265.05 ms - 30.2
OpenCL Interleave 0|1: 221/2215.65 ms - 30.2
```

description:
```
<gpu id>|<thread id on the gpu>: <last delay>/<average calculation per hash bunch> ms - <interleave value>

```
`last delay` should gou slowly to 0.
If it goes down and than jumps to a very large value multiple times within a minute you should reduce the intensity by 5.
The `intensity value` will automatically go up and down within the range of +-5% to adjust kernel run-time fluctuations.
Automatic adjustment is disabled as long as `auto-tuning` is active and will be started after it is finished. 
If `last delay` goes down to 10ms and the messages stops and repeated from time to time with delays up to 15ms you will have already a good value.

### disable comp_mode

`comp_mode` means compatibility mode and removes some checks in compute kernel those takes care that the miner can be used on a wide range of AMD/OpenCL GPU devices.
To avoid miner crashes the `intensity` should be a multiple of `worksize` if `comp_mode` is `false`.

### change the scratchpad memory pattern

By changing `strided_index` to `2` the number of contiguous elements (a 16 byte) for one miner thread can be fine tuned with the option `mem_chunk`.

### Increase Memory Pool

By setting the following environment variables before the miner is started OpenCl allows the miner to more threads.
This variables must be set each time before the miner is started else it could be that the miner can not allocate enough memory and is crashing.

```
export GPU_FORCE_64BIT_PTR=1
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
```

*Note:* Windows user must use `set` instead of `export` to define an environment variable.

### Scratchpad Indexing

The layout of the hash scratchpad memory can be changed for each GPU with the option `strided_index` in `amd.txt`.
Try to change the value from the default `true` to `false`.

## CPU Backend

By default the CPU backend can be tuned in the config file `cpu.txt`

### Choose Value for `low_power_mode`

The optimal value for `low_power_mode` depends on the cache size of your CPU, and the number of threads.

The `low_power_mode` can be set to a number between `1` to `5`. When set to a value `N` greater than `1`, this mode increases the single thread performance by `N` times, but also requires at least `2*N` MB of cache per thread. It can also be set to `false` or `true`. The value `false` is equivalent to `1`, and `true` is equivalent to `2`.

This setting is particularly useful for CPUs with very large cache. For example the Intel Crystal Well Processors are equipped with 128MB L4 cache, enough to run 8 threads at an optimal `low_power_mode` value of `5`.
