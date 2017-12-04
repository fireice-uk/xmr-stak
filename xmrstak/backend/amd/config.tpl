R"===(
/*
 * GPU configuration. You should play around with intensity and worksize as the fastest settings will vary.
 *      index    - GPU index number usually starts from 0
 *  intensity    - Number of parallel GPU threads (nothing to do with CPU threads)
 *   worksize    - Number of local GPU threads (nothing to do with CPU threads)
 * affine_to_cpu - This will affine the thread to a CPU. This can make a GPU miner play along nicer with a CPU miner.
 * strided_index - switch memory pattern used for the scratch pad memory
 *                 true  = use 16byte contiguous memory per thread, the next memory block has offset of intensity blocks
 *                 false = use a contiguous block of memory per thread
 * "gpu_threads_conf" :
 * [
 *	{ "index" : 0, "intensity" : 1000, "worksize" : 8, "affine_to_cpu" : false, "strided_index" : true },
 * ],
 */

"gpu_threads_conf" : [
GPUCONFIG
],

/*
 * Platform index. This will be 0 unless you have different OpenCL platform - eg. AMD and Intel.
 */
"platform_index" : PLATFORMINDEX,

)==="
