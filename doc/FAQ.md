# FAQ

## Content Overview
* [SeLockMemoryPrivilege failed](#selockmemoryprivilege-failed)
* [VirtualAlloc failed](#virtualalloc-failed)
* [Error msvcp140.dll and vcruntime140.dll not available](#error-msvcp140dll-and-vcruntime140dll-not-available)
* [Error: MEMORY ALLOC FAILED: mmap failed](#error-memory-alloc-failed-mmap-failed)
* [Illegal instruction (core dumped)](#illegal-instruction)
* [Virus Protection Alert](#virus-protection-alert)
* [Change Currency to Mine](#change-currency-to-mine)

## SeLockMemoryPrivilege failed

Please see [config.txt](config.txt) under section **LARGE PAGE SUPPORT**

For Windows 7 pro, or Windows 8 and above see [this article](https://msdn.microsoft.com/en-gb/library/ms190730.aspx)  (make sure to reboot afterwards!).

For Windows 7 Home :

1) Download and install [Windows Server 2003 Resource Kit Tools](https://www.microsoft.com/en-us/download/details.aspx?id=17657).  Ignore incompatiablity warning during installation.

2) In cmd or power shell: `ntrights -u %USERNAME% +r SeLockMemoryPrivilege`  (where %USERNAME% is the user that will be running the program.  This command needs to be run as admin)

3) Reboot.

Reference: http://rybkaforum.net/cgi-bin/rybkaforum/topic_show.pl?pid=259791#pid259791

*Warning: do not download ntrights.exe from any other site other then the offical Microsoft download page.*

## VirtualAlloc failed

If you set up the user rights properly (see above), and your system has 4-8GB of RAM (50%+ use), there is a significant chance that there simply won't be a large enough chunk of contiguous memory because Windows is fairly bad at mitigating memory fragmentation.

If that happens, disable all auto-staring applications and run the miner after a reboot.

## Error msvcp140.dll and vcruntime140.dll not available

Download and install this [runtime package](https://go.microsoft.com/fwlink/?LinkId=746572) from Microsoft.  *Warning: Do NOT use "missing dll" sites - dll's are exe files with another name, and it is a fairly safe bet that any dll on a shady site like that will be trojaned.  Please download offical runtimes from Microsoft above.*


## Error: MEMORY ALLOC FAILED: mmap failed

On Linux you will need to configure large page support `sudo sysctl -w vm.nr_hugepages=128` and increase your
ulimit -l. To do this you need to add following lines to /etc/security/limits.conf:

    * soft memlock 262144
    * hard memlock 262144

Save file.  You WILL need to log out and log back in for these settings to take affect on your user (no need to reboot, just relogin in your session).

You can also do it Windows-style and simply run-as-root, but this is NOT recommended for security reasons.

## Illegal Instruction

This typically means you are trying to run it on a CPU that does not have [AES](https://en.wikipedia.org/wiki/AES_instruction_set).  This only happens on older version of miner, new version gives better error message (but still wont' work since your CPU doesn't support the required instructions).

## Virus Protection Alert

Some Virus protection software flag the miner binary as *Male Ware*.
In this case the binary is moved to the quarantine area of the protection software.
This is a wrong alert and not avoid by use.
Add the binary to to protection software white list to solve this issue.s

## Change Currency to Mine

If the miner is compiled for Monero and Aeon than you can change
 - the value `currency` in the config *or*
 - start the miner with the [command line option](usage.md) `--currency monero` or `--currency aeon`
