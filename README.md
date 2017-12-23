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

XMR-Stak 2.2.0 Windows Build Checksums

compiled by: psychocrypt

$ sha1sum *
3f1634244ccd336f7df581e3c82e1c6ca38ce714  libeay32.dll
538f3bd9dfcafc379e912562bcf343333f5375c7  ssleay32.dll
302e5be7c97fcd4922bf99b3533c0523ead5d109  xmrstak_cuda_backend.dll
ad6b9e62a7ea132e1bec0efd8d9e5f8a2ae531ca  xmr-stak.exe
393bc5deb7e59e61cc7f4ccc0f4438402422f3b0  xmrstak_opencl_backend.dll

$ sha3sum *
5aeefca7278be1b2706d99bf89fa23646931f881aff8bbca33654eb1  libeay32.dll
6b696caa620b0c6372881b11e503313152b5191c2d5497b26f81ab79  ssleay32.dll
038de57a707664c7c3ab3a74c8bdb3ed4e22000a74d8b7c359c7c4b5  xmrstak_cuda_backend.dll
19ab61049051178a362dc0d1c17af06f5ca1eb0a75182c0388e5aa22  xmr-stak.exe
cc7ba0fbde50d72df2a530ce52a831578cfa19999841eb954554a022  xmrstak_opencl_backend.dll

date
Fri Dec 22 22:09:59 CET 2017

-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAEBCAAGBQJaPXYSAAoJEAUWOMCIZelDQpAH/As2BD6qDZvbKH5NPHjjDv6T
KBJ6/0h+x2k4Iy3GelrtaogB4LvUDzci4MRfaTXr23Xr+rhwsx3J2xvVdWKZgPXh
bQm5pTJFhiao6Dh+Orway6TLmuaEBLNtknatSkjPUPKmkVd/A7kxxkdelDB//yb+
7k5HGb84T+HU8HBlB00pDITyXv/414egpZGMqWeBXsYDeEYa8KHZlEIO3YI4JrEz
pNW44Q1YcWZ+zxqTDrvMgjW8KJZcXg6ijJ3fEhGBo+hcnF+WuUB3Yd3Frf0ps5J5
MjnWXl/uOobML6K70g2UQcHcEDbPk8f9LUxX1++/I0aHsRMGMYhRj0ad5KYE1IY=
=VCEv
-----END PGP SIGNATURE-----
```
