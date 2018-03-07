:: Windows Command Prompt script for building xmr-stak
:: Author: murlakatamenka (https://github.com/murlakatamenka)
:: Dependencies:
:: - PowerShell (bundled with OS starting from Windows 7): minor dependency
:: - cmake (https://cmake.org/download/): installed and visible via PATH
:: - Visual Studio 2017: MSBuild, VC++ 2017 v141 toolset (https://www.visualstudio.com/downloads/)
:: - CUDA Toolkit (https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
:: - OpenSSL, hwloc, libmicrohttpd (https://github.com/fireice-uk/xmr-stak-dep/releases/download/v2/xmr-stak-dep.zip): official binary build of corresponding xmr-stak dependencies. Unzip to the root folder of the xmr-stak repository.

echo Build started: & powershell get-date -format "{dd-MMM-yyyy HH:mm:ss}"

cd ..\..

pushd .
:: Setting CUDA 9.1 compatible toolset
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.11
@echo on
:: checking compilers versions
cl -Bv

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"
@echo on
popd

:: setting multiple processes compiling
set CL=/MP

set CMAKE_PREFIX_PATH=.\xmr-stak-dep\hwloc;.\xmr-stak-dep\libmicrohttpd;.\xmr-stak-dep\openssl

mkdir build
cd build

cmake -G "Visual Studio 15 2017 Win64" -T v141,host=x64 ..

:: Generic build options
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake .. -DMICROHTTPD_ENABLE=ON
cmake .. -DOpenSSL_ENABLE=ON
:: Other currency options: 'monero' or 'aeon'
cmake .. -DXMR-STAK_CURRENCY=all
cmake .. -DXMR-STAK_COMPILE=native

:: CPU
cmake .. -DCPU_ENABLE=OFF
cmake .. -DHWLOC_ENABLE=ON

:: AMD GPU
cmake .. -DOpenCL_ENABLE=OFF

:: NVIDIA GPU
cmake .. -DCUDA_ENABLE=ON
:: Lists with CUDA compute architectures: https://developer.nvidia.com/cuda-gpus and https://en.wikipedia.org/wiki/CUDA#GPUs_supported
:: Quick reference
:: Architecture: GPUs - corresponding values
:: Kepler: 	GTX 6xx, 7xx 	- 30;35;37
:: Maxwell: GTX 9xx			- 50;52
:: Pascal: 	GTX 10xx		- 60;61;62
:: Volta: 	GTX Titan V		- 70
cmake .. -DCUDA_ARCH=30;35;37;50;52;60;61;62;70
:: set CC=clang
:: set CXX=clang++
:: set .. -DCUDA_COMPILER=clang
cmake .. -DCUDA_COMPILER=nvcc
cmake .. -DXMR-STAK_LARGEGRID=ON
cmake .. -DXMR-STAK_THREADS=0

cmake --build . --config Release --target install

echo Build finished: & powershell get-date -format "{dd-MMM-yyyy HH:mm:ss}"

copy ..\xmr-stak-dep\openssl\bin\*.dll .\bin\Release
cd bin\Release & ren xmr-stak.exe xmr-stak-gpu-nvidia.exe & start .

echo.
pause