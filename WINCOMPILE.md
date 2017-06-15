# Compile **xmr-stak** for Windows

## Install Dependencies

### Preparation

- open a command line `cmd`
- run `mkdir C:\xmr-stak-dep`

### Visual Studio 2017 Community

- download VS2017 Community and install from [https://www.visualstudio.com/downloads/](https://www.visualstudio.com/downloads/)
- during the install chose the components
  - `Desktop development with C++` (left side)
  - `Toolset for Visual Studio C++ 2015.3 v140...` (left side)

### CMake for Win64

- download and install the latest version from [https://cmake.org/download/](https://cmake.org/download/)
- tested version: [cmake 3.9](https://cmake.org/files/v3.9/cmake-3.9.0-rc3-win64-x64.msi)
- during the install choose the option `Add CMake to the system PATH for all users`

### OpenSSL for Win64

- download and install the precompiled binary form [https://slproweb.com/products/Win32OpenSSL.html](https://slproweb.com/products/Win32OpenSSL.html)
- tested version: [OpenSSL 1.0.2L](https://slproweb.com/download/Win64OpenSSL-1_0_2L.exe)

### Hwloc for Win64

- download the precompiled binary from [https://www.open-mpi.org/software/hwloc/v1.11/](https://www.open-mpi.org/software/hwloc/v1.11/)
- tested version: [hwloc-win64-build-1.11.7](https://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-win64-build-1.11.7.zip)
- unzip hwloc to `C:\xmr-stak-dep`

### Microhttpd for Win32

- download the precompiled binary from [http://ftpmirror.gnu.org/libmicrohttpd/](http://ftpmirror.gnu.org/libmicrohttpd/)
- tested version: [microhttpd-0.9.55](http://mirror.reismil.ch/gnu/libmicrohttpd/libmicrohttpd-0.9.55-w32-bin.zip)
- unzip microhttpd to ``C:\xmr-stak-dep`

### Validate the Dependency Folder

- open a command line `cmd`
- run
   ```
   cd c:\xmr-stak-dep
   tree
   ```
- the result should have the same structure
  ```
    C:\XMR-STAK-DEP
    ├───hwloc-win64-build-1.11.7
    │   ├───bin
    │   ├───include
    │   │   └───hwloc
    │   │       └───autogen
    │   ├───lib
    │   │   └───pkgconfig
    │   └───share
    │       ├───doc
    │       │   └───hwloc
    │       ├───hwloc
    │       └───man
    │           ├───man1
    │           ├───man3
    │           └───man7
    └───microhttpd
  ```

## Compile

- download and unzip `xmr-stak-cpu`
- open a command line `cmd`
- goto your unzipped source code
- run (the path to VS2017 can be differ)
  ```
  "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"
  set CMAKE_PREFIX_PATH=C:\xmr-stak-dep\hwloc-win64-build-1.11.7;C:\xmr-stak-dep\microhttpd
  mkdir build
  cd build
  cmake -G "Visual Studio 15 2017 Win64" -T v141,host=x64 ..
  msbuild xmr-stak-cpu.sln /p:Configuration=Release
  cd bin\Release
  copy C:\xmr-stak-dep\hwloc-win64-build-1.11.7\bin\libhwloc-5.dll .
  copy ..\..\..\config.txt .
  ```
- add the pool, pool-password and pool-username to `config.txt`
