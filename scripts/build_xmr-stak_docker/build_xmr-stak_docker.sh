#!/bin/bash -uex

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

if [ -d xmr-stak-cpu ]; then
  git -C xmr-stak-cpu clean -fd
else
  git clone https://github.com/fireice-uk/xmr-stak-cpu.git
fi


########################
# Fedora 26
########################
docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu fedora:26 /bin/bash -c "
set -ex ;
dnf install -y -q gcc gcc-c++ hwloc-devel libmicrohttpd-devel libstdc++-static make openssl-devel cmake ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_fedora_26
git -C xmr-stak-cpu clean -fd


########################
# Ubuntu (17.04)
########################
docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu ubuntu:17.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_ubuntu_17.04
git -C xmr-stak-cpu clean -fd


########################
# Ubuntu 16.04
########################
docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu ubuntu:16.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_ubuntu_16.04
git -C xmr-stak-cpu clean -fd


########################
# Ubuntu 14.04
########################
docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu ubuntu:14.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq curl libmicrohttpd-dev libssl-dev libhwloc-dev software-properties-common ;
add-apt-repository -y ppa:ubuntu-toolchain-r/test ;
apt update -qq ;
apt install -y -qq gcc-7 g++-7 make ;
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1 --slave /usr/bin/g++ g++ /usr/bin/g++-7 ;
curl -L https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz | tar -xzf - -C /tmp/ ;
( cd /tmp/cmake-3.9.0/ && ./configure && make && sudo make install && cd - ) > /dev/null
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_ubuntu_14.04
git -C xmr-stak-cpu clean -fd


########################
# CentOS 7
########################
docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu centos:7 /bin/bash -c "
set -ex ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-4 - << EOF
cd /xmr-stak-cpu ;
cmake3 -DCMAKE_LINK_STATIC=ON . ;
make install ;
EOF
"
mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_centos_7
git -C xmr-stak-cpu clean -fd


########################
# CentOS 6.x
########################
docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu centos:6 /bin/bash -c "
set -ex ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-4 - << EOF
cd /xmr-stak-cpu ;
cmake3 -DCMAKE_LINK_STATIC=ON . ;
make install ;
EOF
"
mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_centos_6
git -C xmr-stak-cpu clean -fd

rm -rf xmr-stak-cpu
