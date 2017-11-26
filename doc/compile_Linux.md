# Compile **xmr-stak** for Linux

## Install Dependencies

### AMD APP SDK 3.0 (only needed to use AMD GPUs)

- download and install the latest version from [http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)

### Cuda 8.0+ (only needed to use NVIDIA GPUs)

- donwload and install [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- for minimal install choose `Custom installation options` during the install and select
    - CUDA/Develpment
    - CUDA/Runtime
    - Driver components

### GNU Compiler
```
    # Ubuntu / Debian
    sudo apt install libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev
    git clone https://github.com/fireice-uk/xmr-stak.git
    mkdir xmr-stak/build
    cd xmr-stak/build
    cmake ..
    make install

    # Arch
    sudo pacman -S base-devel hwloc openssl cmake libmicrohttpd
    git clone https://github.com/fireice-uk/xmr-stak.git
    mkdir xmr-stak/build
    cd xmr-stak/build
    cmake ..
    make install

    # Fedora
    sudo dnf install gcc gcc-c++ hwloc-devel libmicrohttpd-devel libstdc++-static make openssl-devel cmake
    git clone https://github.com/fireice-uk/xmr-stak.git
    mkdir xmr-stak/build
    cd xmr-stak/build
    cmake ..
    make install

    # CentOS
    sudo yum install centos-release-scl epel-release
    sudo yum install cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make
    sudo scl enable devtoolset-4 bash
    git clone https://github.com/fireice-uk/xmr-stak.git
    mkdir xmr-stak/build
    cd xmr-stak/build
    cmake3 ..
    make install

    # Ubuntu 14.04
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install gcc-5 g++-5 make
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 1 --slave /usr/bin/g++ g++ /usr/bin/g++-5
    curl -L http://www.cmake.org/files/v3.4/cmake-3.4.1.tar.gz | tar -xvzf - -C /tmp/
    cd /tmp/cmake-3.4.1/ && ./configure && make && sudo make install && cd -
    sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
    sudo apt install libmicrohttpd-dev libssl-dev libhwloc-dev
    git clone https://github.com/fireice-uk/xmr-stak.git
    mkdir xmr-stak/build
    cd xmr-stak/build
    cmake ..
    make install
```

- g++ version 5.1 or higher is required for full C++11 support. 
If you want to compile the binary without installing libraries / compiler or just compile binary for some other distribution, please check the [build_xmr-stak_docker.sh script](scripts/build_xmr-stak_docker/build_xmr-stak_docker.sh).

### To do a generic and static build for a system without gcc 5.1+
```
    cmake -DCMAKE_LINK_STATIC=ON -DXMR-STAK_COMPILE=generic .
    make install
    cd bin\Release
    copy C:\xmr-stak-dep\openssl\bin\* .
```
Note - cmake caches variables, so if you want to do a dynamic build later you need to specify '-DCMAKE_LINK_STATIC=OFF'
