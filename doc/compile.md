# Compile xmr-stak

## Content Overview
* [Build System](#build-system)
* [Generic Build Options](#generic-build-options)
* [CPU Build Options](#cpu-build-options)
* [AMD Build Options](#amd-build-options)
* [NVIDIA Build Options](#nvidia-build-options)
* [Compile on Windows](compile_Windows.md)
* [Compile on Linux](compile_Linux.md)
* [Compile on FreeBSD](compile_FreeBSD.md)
* [Compile on macOS](compile_macOS.md)

## Build System

The build system is CMake, if you are not familiar with CMake you can learn more [here](https://cmake.org/runningcmake/).

By default the miner will be build with all dependencies. Each optional dependency can be disabled (this will reduce the miner features).

There are two easy ways to set variables for `cmake` to configure *xmr-stak*
- use the ncurses GUI
  - `ccmake ..`
  - edit your options
  - end the GUI by pressing the key `c`(create) and than `g`(generate)
- set Options on the command line
  - enable an option: `cmake .. -DNAME_OF_THE_OPTION=ON`
  - disable an option `cmake .. -DNAME_OF_THE_OPTION=OFF`
  - set a value `cmake .. -DNAME_OF_THE_OPTION=value`

After the configuration you need to compile the miner, follow the guide for your platform:
* [Compile in Windows](compile_Windows.md)
* [Compile in Linux](compile_Linux.md)
* [Compile in FreeBSD](compile_FreeBSD.md)
* [Compile in macOS](compile_macOS.md)

## Generic Build Options
- `CMAKE_INSTALL_PREFIX` install miner to the home folder
  - `cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/xmr-stak`
  - you can find the binary and the `config.txt` file after `make install` in `$HOME/xmr-stak/bin`
- `CMAKE_LINK_STATIC` link libgcc and libstdc++ libraries static (default OFF)
  - disable with `cmake .. -DCMAKE_LINK_STATIC=ON`
  - if you use static compile to run the miner on another system set `-DXMR-STAK_COMPILE=generic`
- `CMAKE_BUILD_TYPE` set the build type
  - valid options: `Release` or `Debug`
  - you should always keep `Release` for your productive miners
- `MICROHTTPD_ENABLE` allows to disable/enable the dependency *microhttpd*
  - there is no *http* interface available if option is disabled: `cmake .. -DMICROHTTPD_ENABLE=OFF`
- `OpenSSL_ENABLE` allows to disable/enable the dependency *OpenSSL*
  - it is not possible to connect to a *https* secured pool if option is disabled: `cmake .. -DOpenSSL_ENABLE=OFF`
- `XMR-STAK_COMPILE` select the CPU compute architecture (default: native)
  - native means the miner binary can be used only on the system where it is compiled but will archive the highest hash rate
  - use `cmake .. -DXMR-STAK_COMPILE=generic` to run the miner on all CPU's with sse2

## CPU Build Options

- `CPU_ENABLE` allows to disable/enable the CPU backend of the miner
- `HWLOC_ENABLE` allows to disable/enable the dependency *hwloc*
  - the config suggestion is not optimal if option is disabled: `cmake .. -DHWLOC_ENABLE=OFF`
  - disabling can be reduce the miner performance

## AMD Build Options

- `OpenCL_ENABLE` allows to disable/enable the AMD backend of the miner

## NVIDIA Build Options

- `CUDA_ENABLE` allows to disable/enable the NVIDIA backend of the miner
- `CUDA_ARCH` build for a certain compute architecture
  - this option needs a semicolon separated list
  - `cmake .. -DCUDA_ARCH=61` or `cmake .. -DCUDA_ARCH=20;61`
  - [list](https://developer.nvidia.com/cuda-gpus) with NVIDIA compute architectures
  - by default the miner is created for all currently available compute architectures
- `CUDA_COMPILER` select the compiler for the device code
  - valid options: `nvcc` or `clang` if clang 3.9+ is installed
```
    # compile host and device code with clang
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    cmake .. -DCUDA_COMPILER=clang
```
- `XMR-STAK_LARGEGRID` use `32` or `64` bit integer for on device indices
  - default is enabled
  - on old GPUs it can increase the hash rate if disabled: `cmake .. -DXMR-STAK_LARGEGRID=OFF`
  - if disabled it is not allowed to use more than `1000` threads on the device
- `XMR-STAK_THREADS` give the compiler information which value for `threads` is used at runtime
  - default is `0` (compile time optimization)
  - if the miner is compiled and used at runtime with the some value it can increase the hash rate: `cmake .. -DXMR-STAK_THREADS=32`
