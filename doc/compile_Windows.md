# Compile **xmr-stak** for Windows

## Install Dependencies

### Preparation

- open a command line `cmd`
- run `mkdir C:\xmr-stak-dep`

### Visual Studio 2017 Community

- download VS2017 Community and install from [https://www.visualstudio.com/downloads/](https://www.visualstudio.com/downloads/)
- during the install chose the components
  - `Desktop development with C++` (left side)
  - `Toolset for Visual Studio C++ 2015.3 v140...` (right side)

### CMake for Win64

- download and install the latest version from [https://cmake.org/download/](https://cmake.org/download/)
- tested version: [cmake 3.9](https://cmake.org/files/v3.9/cmake-3.9.0-rc3-win64-x64.msi)
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
- with CUDA 8: download the version 1 of the precompiled binary from [https://github.com/fireice-uk/xmr-stak-dep/releases/download/v1/xmr-stak-dep.zip](https://github.com/fireice-uk/xmr-stak-dep/releases/download/v1/xmr-stak-dep.zip)
- with CUDA 9: download the version 2 of the precompiled binary from [https://github.com/fireice-uk/xmr-stak-dep/releases/download/v2/xmr-stak-dep.zip](https://github.com/fireice-uk/xmr-stak-dep/releases/download/v2/xmr-stak-dep.zip)
- unzip all to `C:\xmr-stak-dep`

### Validate the Dependency Folder

- open a command line `cmd`
- run
   ```
   cd c:\xmr-stak-dep
   tree .
   ```
- the result should have the same structure
  ```
    C:\xmr-stak-dep>tree .
    Folder PATH listing for volume Windows
    Volume serial number is XX02-XXXX
    C:\XMR-STAK-DEP
    ├───hwloc
    │   ├───include
    │   │   ├───hwloc
    │   │   │   └───autogen
    │   │   └───private
    │   │       └───autogen
    │   └───lib
    ├───libmicrohttpd
    │   ├───include
    │   └───lib
    └───openssl
        ├───bin
        ├───include
        │   └───openssl
        └───lib
  ```

## Compile

- download and unzip `xmr-stak`
- open the command line terminal `cmd`
- `cd` to your unzipped source code directory
- execute the following commands (NOTE: path to VS2017 can be different)
  ```
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
  ```
