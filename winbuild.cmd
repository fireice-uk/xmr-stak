


set CMAKE_PREFIX_PATH=C:\xmr-stak-dep\hwloc
cmake.exe -G  "Visual Studio 14 2015 Win64" -T v140,host=x64 -DMICROHTTPD_ENABLE=OFF -DOpenSSL_ENABLE=OFF  -DOpenCL_ENABLE=ON -DXMR-STAK_COMPILE=generic -DCMAKE_LINK_STATIC=ON .. 
cmake --build . --config Release --target install


