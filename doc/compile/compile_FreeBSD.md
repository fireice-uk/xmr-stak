# Compile **xmr-stak** for FreeBSD

## Install Dependencies

*Note: This guide is tested for FreeBSD 11.0-RELEASE*

From the root shell, run the following commands:

    pkg install git libmicrohttpd hwloc cmake

Type 'y' and hit enter to proceed with installing the packages.

    git clone https://github.com/fireice-uk/xmr-stak.git
    mkdir xmr-stak/build
    cd xmr-stak/build
    cmake ..
    make install

Now you have the binary located at "bin/xmr-stak" and the needed shared libraries.
