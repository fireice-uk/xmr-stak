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


