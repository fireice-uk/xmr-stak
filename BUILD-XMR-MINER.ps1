$path = (Resolve-Path .\).Path;
$currentFolder = @("`$path");

& "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsMSBuildCmd.bat"

& "set CMAKE_PREFIX_PATH=C:\xmr-stak-dep\hwloc;C:\xmr-stak-dep\libmicrohttpd;C:\xmr-stak-dep\openssl"

& "mkdir build"
& "cd build"

& "cmake -G `"Visual Studio 15 2017 Win64`" -T v141,host=x64 .."

& "cmake --build . -DXMR-STAK_CURRENCY=monero -DOpenSSL_ENABLE=OFF -DXMR-STAK_COMPILE=generic --config Release --target install -DXMR-STAK_CURRENCY=monero -DOpenSSL_ENABLE=OFF -DXMR-STAK_COMPILE=generic"

& "cd bin\release"

& "copy C:\xmr-stak-dep\openssl\bin\* .";