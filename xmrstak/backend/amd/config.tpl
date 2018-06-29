R"===(
/*
 * GPU configuration. You should play around with intensity and worksize as the fastest settings will vary.
 * index         - GPU index number usually starts from 0
 * intensity     - Number of parallel GPU threads (nothing to do with CPU threads)
 * worksize      - Number of local GPU threads (nothing to do with CPU threads)
 * affine_to_cpu - This will affine the thread to a CPU. This can make a GPU miner play along nicer with a CPU miner.
 * strided_index - switch memory pattern used for the scratch pad memory
 *                 2 = chunked memory, chunk size is controlled by 'mem_chunk'
 *                     required: intensity must be a multiple of worksize
 *                 1 or true  = use 16byte contiguous memory per thread, the next memory block has offset of intensity blocks
 *                 0 or false = use a contiguous block of memory per thread
 * mem_chunk     - range 0 to 18: set the number of elements (16byte) per chunk
 *                 this value is only used if 'strided_index' == 2
 *                 element count is computed with the equation: 2 to the power of 'mem_chunk' e.g. 4 means a chunk of 16 elements(256byte)
 * comp_mode     - Compatibility enable/disable the automatic guard around compute kernel which allows
 *                 to use a intensity which is not the multiple of the worksize.
 *                 If you set false and the intensity is not multiple of the worksize the miner can crash:
 *                 in this case set the intensity to a multiple of the worksize or activate comp_mode.
 * "gpu_threads_conf" :
 * [
 *	{ "index" : 0, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false, "strided_index" : true, "mem_chunk" : 2, "comp_mode" : true },
 * ],
 * If you do not wish to mine with your AMD GPU(s) then use:
 * "gpu_threads_conf" :
 * null,
 */

"gpu_threads_conf" : [
GPUCONFIG
],

/*
 * Platform index. This will be 0 unless you have different OpenCL platform - eg. AMD and Intel.
 */
"platform_index" : PLATFORMINDEX,

)==="
