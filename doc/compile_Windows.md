# Compile **xmr-stak** for Windows

## Install Build Tools and Dependencies

### Visual Studio 2017 Community

- download VS2017 Community and install from [https://www.visualstudio.com/downloads/](https://www.visualstudio.com/downloads/)
- during the install chose the components
  - `Desktop development with C++` (left side)
  - `VC++ 2015.3 v140 toolset for desktop (x86,x64)` (right side)

### CMake for Win64

- download and install the latest version from [https://cmake.org/download/](https://cmake.org/download/)
- tested version: [cmake 3.9](https://cmake.org/files/v3.9/cmake-3.9.5-win64-x64.msi)
- during the install choose the option `Add CMake to the system PATH for all users`

### Cuda 8.0+ (only needed to use NVIDIA GPUs)

- donwload and install [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- for minimal install choose `Custom installation options` during the install and select
    - CUDA/Develpment
    - CUDA/Visual Studio Integration (ignore the warning during the install that VS2017 is not supported)
    - CUDA/Runtime
    - Driver components

### AMD APP SDK 3.0 (only needed to use AMD GPUs)

- download and install the latest version from [http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)

### Dependencies OpenSSL/Hwloc and Microhttpd

- download the precompiled binary from [https://github.com/fireice-uk/xmr-stak-dep/releases/download/v1/xmr-stak-dep.zip](https://github.com/fireice-uk/xmr-stak-dep/releases/download/v1/xmr-stak-dep.zip)
- unzip all to `C:\xmr-stak-dep`

### Download Source Code

- download [https://codeload.github.com/fireice-uk/xmr-stak/zip/dev](https://codeload.github.com/fireice-uk/xmr-stak/zip/dev)
- unzip to `C:\xmr-stak-dev`

## Compile

- open the command prompt `cmd`
- execute the following commands (NOTE: path to VS2017 can be different)
  ```
  cd C:\xmr-stak-dev\xmr-stak-dev
  "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"
  set CMAKE_PREFIX_PATH=C:\xmr-stak-dep\hwloc;C:\xmr-stak-dep\libmicrohttpd;C:\xmr-stak-dep\openssl
  mkdir build
  cd build
  ```
  - with CUDA 8
    ```
    cmake -G "Visual Studio 15 2017 Win64" -T v140,host=x64 ..
    ```
  - with CUDA 9
    ```
    cmake -G "Visual Studio 15 2017 Win64" -T v141,host=x64 ..
    ```
  ```
  cmake --build . --config Release --target install
  cd bin\Release
  copy C:\xmr-stak-dep\openssl\bin\*.dll .
  ```
