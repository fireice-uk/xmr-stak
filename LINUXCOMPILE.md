# Compile **xmr-stak** for Linux

### GNU Compiler
```
    # Ubuntu / Debian
    sudo apt-get install libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev
    cmake .
    make install

    # Arch
    sudo pacman -S base-devel hwloc openssl cmake libmicrohttpd
    cmake .
    make install

    # Fedora
    sudo dnf install gcc gcc-c++ hwloc-devel libmicrohttpd-devel openssl-devel cmake
    cmake .
    make install

    # CentOS
    sudo yum install centos-release-scl cmake3 hwloc-devel libmicrohttpd-devel openssl-devel
    sudo yum install devtoolset-4-gcc*
    sudo scl enable devtoolset-4 bash
    cmake3 .
    make install
```

- g++ version 5.1 or higher is required for full C++11 support. CMake release compile scripts, as well as CodeBlocks build environment for debug builds is included.

### To do a static build for a system without gcc 5.1+
```
    cmake -DCMAKE_LINK_STATIC=ON .
    make install
```
Note - cmake caches variables, so if you want to do a dynamic build later you need to specify '-DCMAKE_LINK_STATIC=OFF'



