# Compile **xmr-stak** for FreeBSD

## Install Dependencies

*Note: This guide is tested for FreeBSD 11.0-RELEASE*

From the root shell, run the following commands:

    pkg install git libmicrohttpd hwloc cmake 

Type 'y' and hit enter to proceed with installing the packages.

    git clone https://github.com/fireice-uk/xmr-stak-cpu.git
    cd xmr-stak-cpu
    cmake .
    make

Now you have the binary located at "bin/xmr-stak-cpu". Either move this file to your desired location or run "make install" to install it to your path.

You can edit the prebuilt [config.txt](config.txt) file found in the root of the repository or you can make your own. This file is required to run xmr-stak-cpu.
