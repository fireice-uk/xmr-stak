#!/bin/bash -uex

if [ -d xmr-stak-cpu ]; then
  sudo git -C xmr-stak-cpu clean -f
else
  sudo git clone https://github.com/fireice-uk/xmr-stak-cpu.git
fi


########################
# Fedora (latest)
########################
sudo docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu fedora:latest /bin/bash -c "
set -ex ;
dnf install -y -q gcc gcc-c++ hwloc-devel libmicrohttpd-devel libstdc++-static make openssl-devel cmake ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
sudo mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_fedora_latest
sudo git -C xmr-stak-cpu clean -f
exit

########################
# Ubuntu (latest)
########################
sudo docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu ubuntu:latest /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
sudo mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_ubuntu_latest
sudo git -C xmr-stak-cpu clean -f


########################
# Ubuntu 16.04
########################
sudo docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu ubuntu:16.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq libmicrohttpd-dev libssl-dev cmake build-essential libhwloc-dev ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
sudo mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_ubuntu_1604
sudo git -C xmr-stak-cpu clean -f


########################
# Ubuntu 14.04
########################
sudo docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu ubuntu:14.04 /bin/bash -c "
set -ex ;
apt update -qq ;
apt install -y -qq curl libmicrohttpd-dev libssl-dev libhwloc-dev software-properties-common ;
add-apt-repository -y ppa:ubuntu-toolchain-r/test ;
apt update -qq ;
apt install -y -qq gcc-7 g++-7 make ;
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1 --slave /usr/bin/g++ g++ /usr/bin/g++-7 ;
curl -L https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz | tar -xzf - -C /tmp/ ;
cd /tmp/cmake-3.9.0/ && ./configure && make && sudo make install && cd - ;
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force ;
cd /xmr-stak-cpu ;
cmake -DCMAKE_LINK_STATIC=ON . ;
make install ;
"
sudo mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_ubuntu_1404
sudo git -C xmr-stak-cpu clean -f


########################
# CentOS (latest)
########################
sudo docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu centos:latest /bin/bash -c "
set -ex ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-4 - << EOF
cd /xmr-stak-cpu ;
cmake3 -DCMAKE_LINK_STATIC=ON . ;
make install ;
EOF
"
sudo mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_centos_latest
sudo git -C xmr-stak-cpu clean -f


########################
# CentOS 6.x
########################
sudo docker run --rm -it -v $PWD/xmr-stak-cpu:/xmr-stak-cpu centos:6 /bin/bash -c "
set -ex ;
yum install -y -q centos-release-scl epel-release ;
yum install -y -q cmake3 devtoolset-4-gcc* hwloc-devel libmicrohttpd-devel openssl-devel make ;
scl enable devtoolset-4 - << EOF
cd /xmr-stak-cpu ;
cmake3 -DCMAKE_LINK_STATIC=ON . ;
make install ;
EOF
"
sudo mv xmr-stak-cpu/bin/xmr-stak-cpu xmr-stak-cpu_centos_6
sudo git -C xmr-stak-cpu clean -f

sudo rm -rf xmr-stak-cpu
