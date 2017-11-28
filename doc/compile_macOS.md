# Compile **xmr-stak** for macOS (CPU only)

This may be useful for experimenting with this miner or testing pool configurations. You will most likely want to mine on a machine with a GPU.

## Install Dependencies

Requires [Homebrew](https://brew.sh/)

```bash
brew install cmake hwloc libmicrohttpd
```

## Build

```bash
cmake . -DCPU_ENABLE=ON -DHWLOC_ENABLE=ON -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF
```

This will build a binary to `./bin/xmr-stak`.
