"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"

set CMAKE_PREFIX_PATH=C:\xmr-stak-dep\hwloc;C:\xmr-stak-dep\libmicrohttpd;C:\xmr-stak-dep\openssl

mkdir build
cd build

cmake -G "Visual Studio 15 2017 Win64" -T v141,host=x64 ..

pause

cmake --build . --config Release --target install
cd bin\Release

pause