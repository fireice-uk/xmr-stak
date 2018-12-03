#!/bin/bash -uex

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

if [ -d xmr-stak ]; then
  git -C xmr-stak clean -fd
else
  git clone https://github.com/fireice-uk/xmr-stak.git
fi

wget -c https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod a+x cuda_*_linux-run


########################
# Fedora 27
########################
# CUDA is not going to work on Fedora 27 beacuse it only supports these distributions: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
docker run --rm -it -v $PWD:/mnt fedora:27 /bin/bash -c "
set -x ;
dnf install -y -q cmake gcc-c++ hwloc-devel libmicrohttpd-devel libstdc++-static make openssl-devel;
cd /mnt/xmr-stak ;
cmake -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
"

test -d fedora_27 || mkdir fedora_27
mv xmr-stak/bin/* fedora_27
git -C xmr-stak clean -fd


########################
# Ubuntu (17.04)
########################
docker run --rm -it -v $PWD:/mnt ubuntu:17.04 /bin/bash -c "
set -x ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /mnt/xmr-stak ;
/mnt/cuda_*_linux-run --silent --toolkit ;
cmake -DCUDA_ENABLE=ON -DOpenCL_ENABLE=OFF . ;
make ;
"

test -d ubuntu_17.10 || mkdir ubuntu_17.10
mv xmr-stak/bin/* ubuntu_17.10
git -C xmr-stak clean -fd


########################
# Ubuntu 16.04
########################
docker run --rm -it -v $PWD:/mnt ubuntu:16.04 /bin/bash -c "
set -x ;
apt update -qq ;
apt install -y -qq cmake g++ libmicrohttpd-dev libssl-dev libhwloc-dev ;
cd /mnt/xmr-stak ;
/mnt/cuda_*_linux-run --silent --toolkit ;
cmake -DCUDA_ENABLE=ON -DOpenCL_ENABLE=OFF . ;
make ;
"

test -d ubuntu_16.04 || mkdir ubuntu_16.04
mv xmr-stak/bin/* ubuntu_16.04
git -C xmr-stak clean -fd


########################
# Ubuntu 14.04
########################
docker run --rm -it -v $PWD:/mnt ubuntu:14.04 /bin/bash -c "
set -x ;
apt update -qq ;
apt install -y -qq curl libmicrohttpd-dev libssl-dev libhwloc-dev software-properties-common ;
add-apt-repository -y ppa:ubuntu-toolchain-r/test ;
apt update -qq ;
apt install -y -qq gcc-6 g++-6 make ;
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 1 --slave /usr/bin/g++ g++ /usr/bin/g++-6 ;
curl -L https://cmake.org/files/LatestRelease/cmake-3.10.0.tar.gz | tar -xzf - -C /tmp/ ;
( cd /tmp/cmake-*/ && ./configure && make && sudo make install && cd - ) > /dev/null
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force ;
cd /mnt/xmr-stak ;
/mnt/cuda_*_linux-run --silent --toolkit ;
cmake -DCUDA_ENABLE=ON -DOpenCL_ENABLE=OFF . ;
make ;
"

test -d ubuntu_14.04 || mkdir ubuntu_14.04
mv xmr-stak/bin/* ubuntu_14.04
git -C xmr-stak clean -fd


########################
# CentOS 7
########################
# CUDA is not going to work on CentOS/RHEL beacuse it's only support gcc-4 in these distributions: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
docker run --rm -it -v $PWD:/mnt centos:7 /bin/bash -c "
set -x ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-7-gcc* hwloc-devel libmicrohttpd-devel make openssl-devel perl ;
scl enable devtoolset-7 - << EOF
cd /mnt/xmr-stak ;
cmake3 -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
EOF
"

test -d centos_7 || mkdir centos_7
mv xmr-stak/bin/* centos_7
git -C xmr-stak clean -fd


########################
# CentOS 6.x
########################
# CUDA is not going to work on CentOS/RHEL beacuse it's only support gcc-4 in these distributions: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
docker run --rm -it -v $PWD:/mnt centos:6 /bin/bash -c "
set -x ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-7-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-7 - << EOF
cd /mnt/xmr-stak ;
cmake3 -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
EOF
"

test -d centos_6 || mkdir centos_6
mv xmr-stak/bin/* centos_6
git -C xmr-stak clean -fd

rm -rf xmr-stak
