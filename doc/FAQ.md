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
* [Which currency must be chosen if my fork coin is not listed](#which-currency-must-be-chosen-if-my-fork-coin-is-not-listed)
* [Internal compiler error: Killed (program cc1plus)](#internal-compiler-error)

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

If that happens, disable all auto-starting applications and run the miner after a reboot.

## Error msvcp140.dll and vcruntime140.dll not available

Download and install this [runtime package](https://go.microsoft.com/fwlink/?LinkId=746572) from Microsoft.  *Warning: Do NOT use "missing dll" sites - dll's are exe files with another name, and it is a fairly safe bet that any dll on a shady site like that will be trojaned.  Please download offical runtimes from Microsoft above.*


## Error: MEMORY ALLOC FAILED: mmap failed

On Linux you will need to configure large page support and increase your memlock limit (`ulimit -l`).

Never put settings directly into `/etc/sysctl.conf` or `/etc/security/limits.conf` as those are system defaults and can be replaced in upgrades, and custom settings in that file are deprecated in all distros since at least wheezy/trusty (has been illegal in RedHat based distros for longer than that), and will be even more deprecated with systemd (it no longer even reads sysctl.conf, ONLY sysctl.d files, for example - there is a link to the old `/etc/sysctl.conf` for backward compatibility but that can go away at any time).  Also adding to `/etc/rc.local` is extra incorrect, systemd does not even use that file anymore (once the sysvinit compatibility layer is gone, rc.local will no longer work).

To check current settings, run `/sbin/sysctl vm.nr_hugepages ; ulimit -l` as whatever user you will run `xmr-stak` as (example shows bad/low sample defaults):

    $ /sbin/sysctl vm.nr_hugepages ; ulimit -l
    vm.nr_hugepages = 0
    16

To set large page support, add the following lines to `/etc/sysctl.d/60-hugepages.conf`:

    vm.nr_hugepages=128

You WILL need to run `sudo sysctl --system` for these settings to take effect on your system (or reboot).  In some cases (many threads, very large CPU, etc) you may need more than 128 (try 256 if there are still complaints from thread inits)

To increase the memlock (ulimit -l), add following lines to `/etc/security/limits.d/60-memlock.conf`:

    *    - memlock 262144
    root - memlock 262144

You WILL need to log out and log back in for these settings to take effect on your user (no need to reboot, just relogin in your session).
Recheck after completing these steps to validate:

    $ /sbin/sysctl vm.nr_hugepages ; ulimit -l
    vm.nr_hugepages = 128
    262144

You can also do it Windows-style and simply run-as-root, but this is NOT recommended for security reasons.  Also running as root does not properly get around the `ulimit -l` being large enough (and limits `*` does not apply to `root` either, it must be specified explicitly).

## Illegal Instruction

This typically means you are trying to run it on a CPU that does not have [AES](https://en.wikipedia.org/wiki/AES_instruction_set).  This only happens on older version of miner, new version gives better error message (but still wont' work since your CPU doesn't support the required instructions).

## Virus Protection Alert

Some virus protection software flags the miner binary as *malware*. This is a false positive — the software does not contain any malware (and since it is open source, you can verify that yourself!)
If your antivirus software flags **xmr-stak**, it will likely move it to its quarantine area. You may have to whitelist **xmr-stak** in your antivirus.

## Change Currency to Mine

If the miner is compiled for Monero and Aeon than you can change
 - the value `currency` in the config *or*
 - start the miner with the [command line option](usage.md) `--currency monero` or `--currency aeon7`
 - run `xmr-stak --help` to see all supported currencies and algorithms

## How can I mine Monero

Set the value `currency` in `pools.txt` to `monero`.

## Which currency must be chosen if my fork coin is not listed

If your coin you want to mine is not listed please check the documentation of the coin and try to find out if `cryptonight` or `cryptonight-lite` is the used algorithm.
Select one of these generic coin algorithms.

## Internal compiler error

Seeing `g++: internal compiler error: Killed (program cc1plus)` is probably related to not enough RAM to compile. 1 Gb RAM should be enough (it is on clean Ubuntu 16.04).
