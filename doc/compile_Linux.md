# Compile **xmr-stak** for Linux

## Install Dependencies

### AMD Driver (only needed to use AMD GPUs)

- the AMD APP SDK is not longer needed (all is included in the driver package)
- download & unzip the AMD driver: https://www.amd.com/en/support
- run `./amdgpu-pro-install --opencl=legacy,pal` from the unzipped folder
- set the environment variable to opencl `export AMDAPPSDKROOT=/opt/amdgpu-pro/`

**ATTENTION** The linux driver 18.3 creating invalid shares. 
If you have an issue with `invalid shares` please downgrade your driver or switch to ROCm.

For linux also the OpenSource driver ROCm 1.9.X+ is a well working alternative, see https://rocm.github.io/ROCmInstall.html

### Cuda 8.0+ (only needed to use NVIDIA GPUs)

- download and install [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
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
    sudo pacman -S --needed base-devel hwloc openssl cmake libmicrohttpd
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
    scl enable devtoolset-4 bash
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

    # TinyCore Linux 8.x
    # TinyCore is 32-bit only, but there is an x86-64 port, known as "Pure 64,"
    # hosted on the TinyCore home page, and it works well.
    # Beware that huge page support is not enabled in the kernel distributed
    # with Pure 64.  Consider http://wiki.tinycorelinux.net/wiki:custom_kernel
    # Note that as of yet there are no distro packages for microhttpd or hwloc.
    # hwloc is easy enough to install manually though, shown below.
    # Also note that only CPU mining has been tested on this platform, thus the
    # disabling of CUDA and OpenCL shown below.
    tce-load -iw openssl-dev.tcz cmake.tcz make.tcz gcc.tcz git.tcz \
                 glibc_base-dev.tcz linux-4.8.1_api_headers.tcz \
                 glibc_add_lib.tcz
    wget https://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-1.11.8.tar.gz
    tar xzvf hwloc-1.11.8.tar.gz
    cd hwloc-1.11.8
    ./configure --prefix=/usr/local
    make
    sudo make install
    cd ..
    git clone http://github.com/fireice-uk/xmr-stak
    cd xmr-stak
    mkdir build
    cd build
    CC=gcc cmake .. -DCUDA_ENABLE=OFF \
                    -DOpenCL_ENABLE=OFF \
                    -DMICROHTTPD_ENABLE=OFF
    make install
```

- g++ version 5.1 or higher is required for full C++11 support.
If you want to compile the binary without installing libraries / compiler or just compile binary for some other distribution, please check the [build_xmr-stak_docker.sh script](scripts/build_xmr-stak_docker/build_xmr-stak_docker.sh).

- Some newer gcc versions are not supported by CUDA (e.g. Ubuntu 17.10). It will require installing gcc 5 but you can avoid changing defaults.

In that case you can force CUDA to use an older compiler in the following way:
```
cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-5 ..
```

- You need 1 Gb RAM to compile (a bit less might be enough, 512 Mb isn't). 

### To do a generic and static build for a system without gcc 5.1+
```
    cmake -DCMAKE_LINK_STATIC=ON -DXMR-STAK_COMPILE=generic .
    make install
    cd bin\Release
    copy C:\xmr-stak-dep\openssl\bin\* .
```
Note - cmake caches variables, so if you want to do a dynamic build later you need to specify '-DCMAKE_LINK_STATIC=OFF'
