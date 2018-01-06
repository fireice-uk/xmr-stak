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

```shell
brew install hwloc libmicrohttpd gcc openssl cmake
cmake . -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=ON
make install
```

### For CPU-only mining

```shell
brew install hwloc libmicrohttpd gcc openssl cmake
cmake . -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF
make install
```

[All available CMake options](compile.md#cpu-build-options)
