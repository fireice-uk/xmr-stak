# Latest version of ubuntu
FROM ubuntu

# Default git repository
ENV GIT_REPOSITORY https://github.com/fireice-uk/xmr-stak.git
ENV CUDA_URL https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
ENV OPENCL_URL http://pages.cs.wisc.edu/~riccardo/assets/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2

ENV CMAKE_C_COMPILER gcc-5
ENV CMAKE_CXX_COMPILER g++-5
ENV XMRSTAK_CMAKE_FLAGS -DXMR-STAK_COMPILE=generic -DCUDA_ENABLE=ON -DOpenCL_ENABLE=OFF

# Innstall packages
RUN apt-get update \
    && set -x \
    && apt-get install -qq --no-install-recommends -y ca-certificates cmake g++ gcc git libhwloc-dev libmicrohttpd-dev libssl-dev lsb-release wget \
    && wget -q $CUDA_URL \
    && chmod u+x /cuda_*_linux-run \
    && /cuda_*_linux-run --silent --toolkit \
    && wget -q $OPENCL_URL -O - | tar xjf - -C / \
    && /AMD-APP-SDK-*-linux64.sh --keep -- --silent --acceptEULA=yes \
    && . /etc/profile.d/AMDAPPSDK.sh \
    && git clone $GIT_REPOSITORY \
    && cd /xmr-stak \
    && cmake -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} ${XMRSTAK_CMAKE_FLAGS} . \
    && make \
    && cd - \
    && mv /xmr-stak/bin/* /usr/local/bin/ \
    && /AMDAPPSDK*/uninstall.sh \
    && /usr/local/cuda/bin/uninstall_cuda_*.pl > /dev/null \
    && rm -rf /AMD-APP-SDK-*-linux64.sh /AMDAPPSDK* /cuda_*_linux-run /xmr-stak \
    && apt-get purge -y -qq cmake g++ gcc git libhwloc-dev libmicrohttpd-dev libssl-dev lsb-release wget \
    && apt-get clean -qq

VOLUME /mnt

WORKDIR /mnt

ENTRYPOINT ["/usr/local/bin/xmr-stak"]
