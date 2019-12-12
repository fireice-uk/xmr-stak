<table>
    <tr>
        <td align="center"><a href=#><img src="/doc/_img/xmr-stak-rx-btn.png"></a></td>
        <td align="center"><a href=https://github.com/fireice-uk/xmr-stak/tree/master/doc/README.md><img src="/doc/_img/xmr-stak-btn.png"></a></td>
        <td align="center"><a href=https://ragerx.lol><img src="/doc/_img/ragerx-btn.png"></a></td>
    </tr>
</table>

<table>
    <tr>
        <td align="center"><a href=#features-overview><img src="/doc/_img/menu-features.png"></a></td>
        <td align="center"><a href=#supported-coins-and-randomx-variants><img src="/doc/_img/menu-supported-coins.png"></a></td>
        <td align="center"><a href=#Donations><img src="/doc/_img/menu-donations.png"></a></td>
        <td align="center"><a href=#get-miner><img src="/doc/_img/menu-get-miner.png"></a></td>
        <td align="center"><a href=#support-additional-guides-and-feedback><img src="/doc/_img/menu-support.png"></a></td>
    </tr>
</table>

<table>
    <tr>
        <td align="center"><a href=usage.md><img src="/doc/_img/usage.png"></a></td>
        <td align="center"><a href=compile/compile.md><img src="/doc/_img/how-to-compile.png"></a></td>
        <td align="center"><a href=tuning.md><img src="/doc/_img/fine-tuning.png"></a></td>
        <td align="center"><a href=troubleshooting.md><img src="/doc/_img/troubleshooting.png"></a></td>
        <td align="center"><a href=FAQ.md><img src="/doc/_img/faq.png"></a></td>
    </tr>
</table>

## Features overview
[<img src="/doc/_img/features.png">](#)


### Supported coins and RandomX variants
Xmr-Stak-RX supports various variants of RandomX algorithm. Use one of the following options (type this coin alias in either `pool.txt` config file or on startup configuration under `"currency"` parameter and miner will pick it's variant of RandomX algorithm for mining):

| Coin name | Coin alias in config | POW type |
| --- | --- |  --- |
| ArQmA | `Arqma` |  RandomARQ |
| Loki Network | `loki` | RandomXL |
| Monero | `monero` | RandomX |
| Wownero (Monero's testnet) | `wownero` | RandomWOW |


## Donations
[<img src="/doc/_img/fee.png">](#)

## Get Miner
Please note that code is developed on the [dev branch](#), if you want to check out the latest updates, before they are merged on [main branch](#), please refer there. Master branch will always point to a version that we consider stable, so you can download the code by simply typing `git clone https://github.com/fireice-uk/xmr-stak.git -b xmr-stak-rx`  

Also you can find the latest releases, changelog and precompiled binaries on GitHub under [releases](#) section.

If you want to compile the miner from source files, navigate to ["how to compile"](#) section of docs or [xmr-stak forum](#) where you will find the latest step-by-step instructions.


## Start Mining
Miner has 2 ways of initial configuring: simple and advanced. The simple method will prompt user with minimum information. Required answers are y , (or yes), n , (or no):

#### Simple setup:
* `Use simple setup method?` y    
* `Please enter the currency that you want to mine:` Enter currency or mining algorithm  
* `Enter pool address (pool address:port):` Enter pool connection address:port  
* `Username (wallet address or pool login):` Enter wallet address
* `Password (mostly empty or x):` press Enter  
* `Does this pool port support TLS/SSL? Use no if unknown. (y/N):` press y or n  

#### Advanced setup:
* `Use simple setup method?` n  
* `Do you want to use the HTTP interface? Unlike the screen display, browser interface is not affected by the GPU lag. If you don't want to use it, please enter 0, otherwise enter port number that the miner should listen on` 5656
* `Please enter the currency that you want to mine:` Enter currency or mining algorithm
* `Enter pool address (pool address:port):` Enter pool connection address:port 
* `Username (wallet address or pool login):` Enter wallet address
* `Password (mostly empty or x):` press Enter
* `Rig identifier for pool-side statistics (needs pool support). Can be empty:` Enter rig name or press Enter
* `Does this pool port support TLS/SSL? Use no if unknown. (y/N)` Enter y or n
* `Do you want to use nicehash on this pool? (y/N)` n
* `Do you want to use multiple pools? (y/N)` Enter y if you want to se up backup pool or n


## Support additional guides and feedback
[<img src="/doc/_img/YT.png">](https://www.youtube.com/c/xmrstak)
To improve our support we created [Xmr-Stak forum](https://www.reddit.com/r/XmrStak) which is also applicable to Xmr-Stak-RX. Check it out if you have a problem, or you are looking for most up to date config for your card and [guides](https://www.reddit.com/r/XmrStak/wiki/index).

<table>
    <tr>
        <td align="center"><a href=usage.md><img src="/doc/_img/usage.png"></a></td>
        <td align="center"><a href=compile/compile.md><img src="/doc/_img/how-to-compile.png"></a></td>
        <td align="center"><a href=tuning.md><img src="/doc/_img/fine-tuning.png"></a></td>
        <td align="center"><a href=troubleshooting.md><img src="/doc/_img/troubleshooting.png"></a></td>
        <td align="center"><a href=FAQ.md><img src="/doc/_img/faq.png"></a></td>
    </tr>
</table>
