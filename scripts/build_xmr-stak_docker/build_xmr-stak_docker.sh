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


########################
# Fedora 27
########################
docker run --rm -it -v $PWD/xmr-stak:/xmr-stak fedora:27 /bin/bash -c "
set -ex ;
dnf install -y -q gcc gcc-c++ hwloc-devel libmicrohttpd-devel libstdc++-static make openssl-devel cmake ;
cd /xmr-stak ;
cmake -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
"

mv xmr-stak/bin/xmr-stak xmr-stak_fedora_27
git -C xmr-stak clean -fd


########################
# Ubuntu (17.04)
########################
docker run --rm -it -v $PWD/xmr-stak:/xmr-stak ubuntu:17.10 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /xmr-stak ;
cmake -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
"

mv xmr-stak/bin/xmr-stak xmr-stak_ubuntu_17.10
git -C xmr-stak clean -fd


########################
# Ubuntu 16.04
########################
docker run --rm -it -v $PWD/xmr-stak:/xmr-stak ubuntu:16.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /xmr-stak ;
cmake -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
"

mv xmr-stak/bin/xmr-stak xmr-stak_ubuntu_16.04
git -C xmr-stak clean -fd


########################
# Ubuntu 14.04
########################
docker run --rm -it -v $PWD/xmr-stak:/xmr-stak ubuntu:14.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq curl libmicrohttpd-dev libssl-dev libhwloc-dev software-properties-common ;
add-apt-repository -y ppa:ubuntu-toolchain-r/test ;
apt update -qq ;
apt install -y -qq gcc-7 g++-7 make ;
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1 --slave /usr/bin/g++ g++ /usr/bin/g++-7 ;
curl -L https://cmake.org/files/LatestRelease/cmake-3.10.0.tar.gz | tar -xzf - -C /tmp/ ;
( cd /tmp/cmake-*/ && ./configure && make && sudo make install && cd - ) > /dev/null
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force ;
cd /xmr-stak ;
cmake -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
"

mv xmr-stak/bin/xmr-stak xmr-stak_ubuntu_14.04
git -C xmr-stak clean -fd


########################
# CentOS 7
########################
docker run --rm -it -v $PWD/xmr-stak:/xmr-stak centos:7 /bin/bash -c "
set -ex ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-4 - << EOF
cd /xmr-stak ;
cmake3 -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
EOF
"

mv xmr-stak/bin/xmr-stak xmr-stak_centos_7
git -C xmr-stak clean -fd


########################
# CentOS 6.x
########################
docker run --rm -it -v $PWD/xmr-stak:/xmr-stak centos:6 /bin/bash -c "
set -ex ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-4 - << EOF
cd /xmr-stak ;
cmake3 -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF . ;
make ;
EOF
"

mv xmr-stak/bin/xmr-stak xmr-stak_centos_6
git -C xmr-stak clean -fd

rm -rf xmr-stak
