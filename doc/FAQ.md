# FAQ

## Content Overview
* ["Obtaining SeLockMemoryPrivilege failed."](#obtaining-selockmemoryprivilege-failed)
* [VirtualAlloc failed](#virtualalloc-failed)
* [Error msvcp140.dll and vcruntime140.dll not available](#error-msvcp140dll-and-vcruntime140dll-not-available)
* [Error: MEMORY ALLOC FAILED: mmap failed](#error-memory-alloc-failed-mmap-failed)
* [Illegal instruction (core dumped)](#illegal-instruction)
* [Virus Protection Alert](#virus-protection-alert)
* [Change Currency to Mine](#change-currency-to-mine)
* [How can I mine Monero](#how-can-i-mine-monero)
* [Why is Monero named monero7](why-is-monero-named-monero7)
* [Which currency must be chosen if my fork coin is not listed](#which-currency-must-be-chosen-if-my-fork-coin-is-not-listed)

## "Obtaining SeLockMemoryPrivilege failed."

For professional versions of Windows see [this article](https://msdn.microsoft.com/en-gb/library/ms190730.aspx).
Make sure to reboot afterwards!

For Windows 7/10 Home:

1) Download and install [Windows Server 2003 Resource Kit Tools](https://www.microsoft.com/en-us/download/details.aspx?id=17657). Ignore any incompatibility warning during installation.

2) Open cmd or PowerShell as an administrator.

3) Use `ntrights -u %USERNAME% +r SeLockMemoryPrivilege` where %USERNAME% is the user that will be running the program.

4) Reboot.

Reference: http://rybkaforum.net/cgi-bin/rybkaforum/topic_show.pl?pid=259791#pid259791

*Warning: Do not download ntrights.exe from any other site other than the offical Microsoft download page.*

## VirtualAlloc failed

If you set up the user rights properly ([see above](https://github.com/fireice-uk/xmr-stak/blob/master/doc/FAQ.md#selockmemoryprivilege-failed)), and your system has 4-8GB of RAM (50%+ use), there is a significant chance that there simply won't be a large enough chunk of contiguous memory because Windows is fairly bad at mitigating memory fragmentation.

If that happens, disable all auto-staring applications and run the miner after a reboot.

## Error msvcp140.dll and vcruntime140.dll not available

Download and install this [runtime package](https://go.microsoft.com/fwlink/?LinkId=746572) from Microsoft.  *Warning: Do NOT use "missing dll" sites - dll's are exe files with another name, and it is a fairly safe bet that any dll on a shady site like that will be trojaned.  Please download offical runtimes from Microsoft above.*


## Error: MEMORY ALLOC FAILED: mmap failed

On Linux you will need to configure large page support and increase your ulimit -l. 

To set large page support, add the following lines to /etc/sysctl.conf:
    
    vm.nr_hugepages=128

To increase the ulimit, add following lines to /etc/security/limits.conf:

    * soft memlock 262144
    * hard memlock 262144

You WILL need to log out and log back in for these settings to take affect on your user (no need to reboot, just relogin in your session).

You can also do it Windows-style and simply run-as-root, but this is NOT recommended for security reasons.

## Illegal Instruction

This typically means you are trying to run it on a CPU that does not have [AES](https://en.wikipedia.org/wiki/AES_instruction_set).  This only happens on older version of miner, new version gives better error message (but still wont' work since your CPU doesn't support the required instructions).

## Virus Protection Alert

Some virus protection software flags the miner binary as *malware*. This is a false positive — the software does not contain any malware (and since it is open source, you can verify that yourself!)
If your antivirus software flags **xmr-stak**, it will likely move it to its quarantine area. You may have to whitelist **xmr-stak** in your antivirus.

## Change Currency to Mine

If the miner is compiled for Monero and Aeon than you can change
 - the value `currency` in the config *or*
 - start the miner with the [command line option](usage.md) `--currency monero7` or `--currency aeon7`
 - run `xmr-stak --help` to see all supported currencies and algorithms

## How can I mine Monero

Set the value `currency` in `pools.txt` to `monero7`.

## Why is Monero named monero7

To avoid configuration conflicts after the hard fork of Monero to the new POW with our old naming schema where all cryptonight currencies was selected by choosing `monero` as currency we decided to switch to the name `monero7`.

## Which currency must be chosen if my fork coin is not listed

If your coin you want to mine is not listed please check the documentation of the coin and try to find out if `cryptonight` or `cryptonight-lite` is the used algorithm.
Select one of these generic coin algorithms.
