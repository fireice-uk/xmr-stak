# Compile **xmr-stak** for MacOS

## Dependencies

Assuming you already have [Homebrew](https://brew.sh) installed, the installation of dependencies is pretty straightforward and will generate the `xmr-stak` binary in the `bin/` directory.

### For NVIDIA GPUs

```shell
brew tap caskroom/drivers
brew cask install nvidia-cuda
brew install hwloc libmicrohttpd gcc openssl cmake
cmake . -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DOpenCL_ENABLE=OFF
make install
```

[All available CMake options](compile.md#nvidia-build-options)

### For AMD GPUs

First download the AMD SDK Installer (64bit) for Linux from here: https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/

Exract the archive then `cd </path/to/extracted/amd-sdk>`

__Next what follows is customizing the installation script to work for MacOS.__

Extract the install to a temporary directory (noted here by 'temp-dir') this keeps the install file so you can edit it:
`./amdinstall.sh --target temp-dir --keep --nox11`

Make the following changes to the install.sh script in the created `temp-dir/` using your editor of choice.
(these are basically find & replace commands, `sed` or similar will work just fine)

* Find every __mkdir__ command and replace the `--mode` option with `-m`
* Find the __cp__ command and replace the `-rvu` option with `-Rv`

Now run the `install.sh` file - it will prompt you with a EULA and options for install location:

```shell
sh ./install.sh
```

If it installed correctly, you should now have the following folders in your install directory of choice:

* bin
* docs
* etc
* include
* lib

The environment variables have also been added to your `.bashrc` file, of course copy those into your shell's _rc_ file of choice if you use someething else.

__Compile xmr-stak__

```shell
cmake . -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DOpenCL_ENABLE=ON -DCUDA_ENABLE=OFF -DHWLOC_ENABLE=OFF
make install
```

__Conclusion__

Set your options in amd.txt and your pool/wallet info in config.txt, then run with:

```shell
xmr-stak --amd amd.txt
```

My results using an intensity of 500, workersize of 8 (iMac AMD R9 M395 2048 MB) - But this is very slow to respond, and unusable during mining. Your milage may vary.

```shell
HASHRATE REPORT - AMD
| ID |    10s |    60s |    15m |
|  0 |  318.8 |   (na) |   (na) |
---------------------------
```


### For CPU-only mining

```shell
brew install hwloc libmicrohttpd gcc openssl cmake
cmake . -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF
make install
```

[All available CMake options](compile.md#cpu-build-options)
