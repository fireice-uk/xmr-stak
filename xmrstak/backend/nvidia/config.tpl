R"===(
/*
 * GPU configuration. You should play around with threads and blocks as the fastest settings will vary.
 * index         - GPU index number usually starts from 0.
 * threads       - Number of GPU threads (nothing to do with CPU threads).
 * blocks        - Number of GPU blocks (nothing to do with CPU threads).
 * bfactor       - Enables running the Cryptonight kernel in smaller pieces.
 *                 Increase if you want to reduce GPU lag. Recommended setting on GUI systems - 8
 * bsleep        - Insert a delay of X microseconds between kernel launches.
 *                 Increase if you want to reduce GPU lag. Recommended setting on GUI systems - 100
 * affine_to_cpu - This will affine the thread to a CPU. This can make a GPU miner play along nicer with a CPU miner.
 * sync_mode     - method used to synchronize the device
 *                 documentation: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac
 *                 0 = cudaDeviceScheduleAuto
 *                 1 = cudaDeviceScheduleSpin - create a high load on one cpu thread per gpu
 *                 2 = cudaDeviceScheduleYield
 *                 3 = cudaDeviceScheduleBlockingSync (default)
 *
 * On the first run the miner will look at your system and suggest a basic configuration that will work,
 * you can try to tweak it from there to get the best performance.
 *
 * A filled out configuration should look like this:
 * "gpu_threads_conf" :
 * [
 *     { "index" : 0, "threads" : 17, "blocks" : 60, "bfactor" : 0, "bsleep" :  0,
 *       "affine_to_cpu" : false, "sync_mode" : 3,
 *     },
 * ],
 * If you do not wish to mine with your nVidia GPU(s) then use:
 * "gpu_threads_conf" :
 * null,
 */

"gpu_threads_conf" :
[
GPUCONFIG
],

)==="
