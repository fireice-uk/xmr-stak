# Compile **xmr-stak** for MacOS

Assuming you already have [Homebrew](https://brew.sh) installed, the compilation for **CPU-only mining** is pretty straightforward:

```
brew install hwloc libmicrohttpd gcc openssl cmake
cmake . -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF
make install
```

The `xmr-stak` binary will be generated in the `bin/` directory.
