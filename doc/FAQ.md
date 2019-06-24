# FAQ
To improve our support we created [Xmr-Stak forum](https://www.reddit.com/r/XmrStak). Check it out if you have a problem, or you are looking for most up to date config for your card and [guides](https://www.reddit.com/r/XmrStak/wiki/index).


## Content Overview
* [Virus Protection Alert](#virus-protection-alert)
* [Change Currency to Mine](#change-currency-to-mine)
* [How can I mine Monero](#how-can-i-mine-monero)
* [Which currency must be chosen if my fork coin is not listed](#which-currency-must-be-chosen-if-my-fork-coin-is-not-listed)

### Virus Protection Alert
Some virus protection software flags the miner binary as *malware*. This is a false positive — the software does not contain any malware (and since it is open source, you can verify that yourself!)
If your antivirus software flags **xmr-stak**, it will likely move it to its quarantine area. You may have to whitelist **xmr-stak** in your antivirus.

### Change Currency to Mine
If the miner is compiled for Monero and Aeon than you can change
 - the value `currency` in the config *or*
 - start the miner with the [command line option](usage.md) `--currency monero` or `--currency aeon7`
 - run `xmr-stak --help` to see all supported currencies and algorithms

### How can I mine Monero
Set the value `currency` in `pools.txt` to `monero`.

### Which currency must be chosen if my fork coin is not listed
If your coin you want to mine is not listed please check the documentation of the coin and try to find out if `cryptonight` or `cryptonight-lite` is the used algorithm.
Select one of these generic coin algorithms.

