R"===(

/* 
 * Number of GPUs that you have in your system. Each GPU will get its own CPU thread.
 */
"gpu_thread_num" : NUMGPUS,

/*
 * GPU configuration. You should play around with intensity and worksize as the fastest settings will vary.
 *      index    - GPU index number usually starts from 0
 *  intensity    - Number of parallel GPU threads (nothing to do with CPU threads)
 *   worksize    - Number of local GPU threads (nothing to do with CPU threads)
 * affine_to_cpu - This will affine the thread to a CPU. This can make a GPU miner play along nicer with a CPU miner.
 */


GPUCONFIG

/*
 * Platform index. This will be 0 unless you have different OpenCL platform - eg. AMD and Intel.
 */
"platform_index" : PLATFORMINDEX,

)==="
