###### fireice-uk's and psychocrypt's
# XMR-Stak-RX: Free RandomX Mining Software

XMR-Stak-RX is a universal free and open source stratum pool miner. This miner supports x86-64 CPUs, AMD and NVIDIA GPUs and can be used for various randomX variants.

## Video guides
[<img src="doc/_img/stak-yt-cover.jpg">](https://www.youtube.com/playlist?list=PLAhUkom29iGMFoN8pk91JA-oqvxlmJ5H8)
###### Video by Crypto Sewer

## Overview
* [Features](#features)
* [Supported coins and algorithms](#supported-coins-and-algorithms)
* [Download](#download)
* [FAQ](doc/FAQ.md)
* [Developer Donation](#developer-donation)
* [Developer PGP Key's](doc/pgp_keys.md)

## Guides and FAQ
To improve our support we created [Xmr-Stak forum](https://www.reddit.com/r/XmrStak). Check it out if you have a problem, or you are looking for most up to date config for your card and [guides](https://www.reddit.com/r/XmrStak/wiki/index).
* [Usage](doc/usage.md)
* [How to compile](doc/compile/compile.md)
* [Fine tuning](doc/tuning.md)
* [FAQ](doc/FAQ.md)
* [Troubleshooting](doc/troubleshooting.md) (Fixing common problems)

## Features

- Supports all common backends (CPU/x86, AMD/NVIDIA GPU).
- Supports all common OS (Linux, Windows and macOS).
- Supports `randomx`, `randomx_loki` and `randomx_wow`
- Easy to use and flexible in setup:
  - guided start with easy/advanced setup option (no need to edit a config file for the first start)
  - auto-configuration and config file creation for each backend.
- Open source software (GPLv3)
- TLS support.
- [HTML statistics](doc/usage.md#html-and-json-api-report-configuraton)
- [JSON API for monitoring](doc/usage.md#html-and-json-api-report-configuraton)

## HTML reports
  <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-hashrate.png" width="260"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-results.png" width="260"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-connection.png" width="260">

## Supported coins and algorithms

Following coins can be mined using this miner:

- [Loki]()
- [Monero](https://getmonero.org)
- [WOW]()


**[Ryo Currency](https://ryo-currency.com)** - is a way for us to implement the ideas that we were unable to in
Monero. See [here](https://github.com/fireice-uk/cryptonote-speedup-demo/) for details.

If your preferred coin is not listed, you can choose one of the following mining algorithms:
    
- 1MiB scratchpad memory
    - randomx_wow
    

- 2MiB scratchpad memory
    - randomx
    - randomx_loki
    
Please note, this list is not complete and is not an endorsement.

## Download

You can find the latest releases and precompiled binaries on GitHub under [Releases](https://github.com/fireice-uk/xmr-stak/releases).

## Developer Donation

If you want to donate directly to support further development, here are our wallets

fireice-uk:
```
4581HhZkQHgZrZjKeCfCJxZff9E3xCgHGF25zABZz7oR71TnbbgiS7sK9jveE6Dx6uMs2LwszDuvQJgRZQotdpHt1fTdDhk
```

psychocrypt:
```
45tcqnJMgd3VqeTznNotiNj4G9PQoK67TGRiHyj6EYSZ31NUbAfs9XdiU5squmZb717iHJLxZv3KfEw8jCYGL5wa19yrVCn
```
