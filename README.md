###### fireice-uk's and psychocrypt's
# XMR-Stak - Monero/Aeon All-in-One Mining Software

XMR-Stak is a universal Stratum pool miner. This miner supports CPUs, AMD and NVIDIA gpus and can be used to mine the crypto currency Monero and Aeon.

## HTML reports
<img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-hashrate.png" width="260"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-results.png" width="260"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-connection.png" width="260">

## Overview
* [Features](#features)
* [Supported altcoins](#supported-altcoins)
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

## Supported altcoins

Besides Monero, following coins can be mined using this miner:

- [Aeon](http://www.aeon.cash/)
- [Electroneum](https://electroneum.com)
- [Intense](https://intensecoin.com)
- [Sumokoin](https://www.sumokoin.org)

For all coins, except Aeon, you can use Monero settings.

Please note, this list is not complete, and is not an endorsement.

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

Please use the [Developer PGP Key's](doc/pgp_keys.md) to verify the integrity of the precompiled binaries.

```
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

XMR-Stak 2.1.0 Windows Build Checksums

compiled by: psychocrypt

$ sha1sum *
3f1634244ccd336f7df581e3c82e1c6ca38ce714  libeay32.dll
538f3bd9dfcafc379e912562bcf343333f5375c7  ssleay32.dll
152042d47afaf9a42d4330440c62d81082fc8e4e  xmrstak_cuda_backend.dll
0effc72e21382d22a0b8221ee55e470093b39715  xmr-stak.exe
3b6ea8d155e89dc876bf6bb436d6826e33a62955  xmrstak_opencl_backend.dll

$ sha3sum *
5aeefca7278be1b2706d99bf89fa23646931f881aff8bbca33654eb1  libeay32.dll
6b696caa620b0c6372881b11e503313152b5191c2d5497b26f81ab79  ssleay32.dll
3a0079c2e4f303a48c4c94817ace2c6de077b099c08d3e2e25d206f2  xmrstak_cuda_backend.dll
1623c54b05329dcd08e477cf9ba750c44a246227359cff42b1c5bb4d  xmr-stak.exe
4826eb52a346e1ec5d979c745a4d642f13dd76e2a9a4b2f9d4bd149a  xmrstak_opencl_backend.dll

date
Sat Dec  9 13:10:01 CET 2017
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAEBCAAGBQJaK9N7AAoJEAUWOMCIZelDBesH/11/YJYV/a1545QcKlTVrSJS
S51nGVlhn1Opi3FIUadaHf+INqJgQmE6+8PSaWo74WdX1TCaCgwszqI+o4EYEKZu
/+/Lmc19++WCFSIV6RozEG/7bGuRO+R+xstm1yh/5Y3DrxJrZFq2fmRu/sodryz9
Iw8tBcUfZUz+M8OepeMUfmu3wqzbOEAJLEw2OSPwkHACTpVFc2n3MWMDxqrRb3GU
YjZrxSMEGU/viSz88uGovbqVU53Ala6jCvqunDcibZ6BoXbSI4qgyUjCcc+uxm1k
xnzF5fgwWHuXH4l3CXQcU/2y6I5in+rNvWT0/pMNSRp5kRDu0SSLYLK/FIIFhNQ=
=KNpx
-----END PGP SIGNATURE-----
```
