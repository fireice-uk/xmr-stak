###### fireice-uk's and psychocrypt's
# XMR-Stak - Monero/Aeon All-in-One Mining Software

XMR-Stak is a universal Stratum pool miner. This miner supports CPUs, AMD and NVIDIA gpus and can be used to mine the crypto currency Monero and Aeon.

## HTML reports
<img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-hashrate.png" width="260"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-results.png" width="260"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-connection.png" width="260">

## Overview
* [Features](#features)
* [Download](#download)
* [Linux Portable Binary](doc/Linux_deployment.md)
* [Usage](doc/usage.md)
* [HowTo Compile](doc/compile.md)
* [FAQ](doc/FAQ.md)
* [Developer Donation](#default-developer-donation)
* [Release Cheksums](#release-checksums)
* [Developer PGP Key's](doc/pgp_keys.md)

## Features

- support all common backends (CPU/x86, AMD-GPU and NVIDIA-GPU)
- support all common OS (Linux, Windows and MacOS)
- supports algorithm cryptonight for Monero (XMR) and cryptonight-light (AEON)
- easy to use
  - guided start (no need to edit a config file for the first start)
  - auto configuration for each backend
- open source software (GPLv3)
- TLS support
- HTML statistics
- JSON API for monitoring

## Download

You can find the latest releases and precompiled binaries on GitHub under [Releases](https://github.com/fireice-uk/xmr-stak/releases).
If you are running on Linux (especially Linux VMs), checkout [Linux Portable Binary](doc/Linux_deployment.md).

## Default Developer Donation

By default the miner will donate 2% of the hashpower (2 minute in 100 minutes) to my pool. If you want to change that, edit [donate-level.hpp](xmrstak/donate-level.hpp) before you build the binaries.

If you want to donate directly to support further development, here is my wallet

fireice-uk:
```
4581HhZkQHgZrZjKeCfCJxZff9E3xCgHGF25zABZz7oR71TnbbgiS7sK9jveE6Dx6uMs2LwszDuvQJgRZQotdpHt1fTdDhk
```

psychocrypt:
```
43NoJVEXo21hGZ6tDG6Z3g4qimiGdJPE6GRxAmiWwm26gwr62Lqo7zRiCJFSBmbkwTGNuuES9ES5TgaVHceuYc4Y75txCTU
```

## Release Checksums
```
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

XMR-Stak 2.0.0 Windows Build Checksums

$ sha1sum *
316aafce7e9e9b9ac76a3c559d2df2ce0d8e0f0e  libeay32.dll
fd5251b1b9a44be590eed18fd6563661c662e095  ssleay32.dll
9c81da9334db8ade7255bf201a2f90c571f09969  xmrstak_cuda_backend.dll
b41b1baea01189c6acfdb03b66b332ae615519b0  xmr-stak.exe
0d90f03886af567100cd0ab780f7a50557e3dafa  xmrstak_opencl_backend.dll

$ sha3sum *
627b5f8c7b67e45b1ff1e344ba841ee45cbab78d03f35bc572a53e77  libeay32.dll
1f587006e26d0a6e0969d0f562ebf622783e3898a0239e4bc08d129f  ssleay32.dll
e9cd4a682a208a68c012635924dc320d4da1b222d358ca21e326f30d  xmrstak_cuda_backend.dll
37b4891b43694548ee3ba83174985b8300a30808fd5e469f6273c0b7  xmr-stak.exe
b574a18d3f729123dfc4efe857021e0d3b48e428562d340acc5de426  xmrstak_opencl_backend.dll

$ date
Sun 19 Nov 16:54:14 GMT 2017
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAEBCAAGBQJaEberAAoJEPsk95p+1Bw012QH/A+c1M1+Jt4tHvSrgYDgMt/g
i/r4ZaCUj8Q43Q/PRo1V5ZwGAjPQp23qMp1b2PX+B9nRZD61uXN4+LX6BJuK8Cvp
eDS3weWHYP0OJHXdSr/2u1VbL09u3att7BhXj8N5Y1k/DnyXtxIFafDZb6rOwOyu
r0iUMMjFNrQBJe3RdrGLeGTc1atxVTnLsa5TmBT8NZTIVk9tfpEzCcyvvfwKuK5T
fjzzOR4m0HcbzOxIqydOLWXkX1oOTHjq1TSCuZ+W1vcp8drCtlY5zM5ckYDD4818
cvX6gTCsFYfLw/p+sz+DN7kh7zJlCvIFga3HaFByxCSuyMY08qerXS/0862ZMdo=
=Gf/1
-----END PGP SIGNATURE-----

```
