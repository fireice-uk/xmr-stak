# Troubleshooting
To improve our support we created [Xmr-Stak forum](https://www.reddit.com/r/XmrStak). Check it out if you have a problem, or you are looking for most up to date config for your card and [guides](https://www.reddit.com/r/XmrStak/wiki/index).


### 1. CL_MEM_OBJECT_ALLOCATION_FAILURE when calling clEnqueue
This error means that GPU can't allocate the requested amount of memory that is specified by your config. There is 2 known solutions of this problem:

* Check if you occasionally use too many threads per one GPU (check *index* value in amd.txt)
* You set too high `intensity` value in amd.txt - try to reduce it to lower values (multiple to `worksize`)
* If you are using Windows - you may have not enough virtual memory in system. Add virtual memory (don't be afraid if it goes up to 60gb per 6 GPU rig)

 

### 2. GPU is not detected
Check if you have antivirus software turned on. If yes - it could delete some .dll files (for example  xmrstak\_cuda\_backend\_cuda10\_0.dll)

 

### 3. Illegal Instruction
This typically means you are trying to run it on a CPU that does not have [AES](https://en.wikipedia.org/wiki/AES_instruction_set). This only happens on older version of miner, new version gives better error message (but still wont' work since your CPU doesn't support the required instructions).

 

### 4.  Internal compiler error
Seeing  `g++: internal compiler error: Killed (program cc1plus)`is probably related to not enough RAM to compile. 1 Gb RAM should be enough (on clean Ubuntu 16.04).

 

### 5. Invalid Result GPU ID
This error can be caused by several reasons, here is most common, known successful practices how to fix it:

* **Hardware problem: overclock/overvoltage/undervoltage** \- try to use stock clocks and voltages.
* **Software problem: drivers** \- try to change driver versions (for AMD gpu most commonly stable versions are: blockchain drivers or 18.6.1)
* **Miner misconfiguration** \- try to reduce `intensity` (if AMD) or `threads` or `bfactor` (if NVIDIA) in config file.

If you still receive these errors, [report please the issue](https://github.com/fireice-uk/xmr-stak/issues).

 
### 6. IP is banned
Pool has banned your IP, This can be caused by several reasons:

* You selected wrong pool port or the static diff is too low. (Learn more about [pool ports and diff](https://www.reddit.com/r/XmrStak/wiki/guides/other-questions#wiki_1._pool_ports_and_difficulty))
* You had too many [invalid shares \[8\]](https://www.reddit.com/r/XmrStak/wiki/troubleshooting#wiki_8._invalid_result_gpu_id)

 

### 7. MEMORY ALLOC FAILED: mmap failed
On Linux you will need to configure large page support and increase your memlock limit (`ulimit -l`).

Never put settings directly into `/etc/sysctl.conf` or `/etc/security/limits.conf`  as those are system defaults and can be replaced in upgrades, and custom settings in that file are deprecated in all distros since at least wheezy/trusty (has been illegal in RedHat based distros for longer than that), and will be even more deprecated with systemd (it no longer even reads sysctl.conf, ONLY sysctl.d files, for example - there is a link to the old `/etc/sysctl.conf` for backward compatibility but that can go away at any time). Also adding to `/etc/rc.local` is extra incorrect, systemd does not even use that file anymore (once the sysvinit compatibility layer is gone, rc.local will no longer work). To check current settings, run `/sbin/sysctl vm.nr_hugepages ; ulimit -l` as whatever user you will run xmr-stak  as (example shows bad/low sample defaults):

    $ /sbin/sysctl vm.nr_hugepages ; ulimit -l vm.nr_hugepages = 0 16 

To set large page support, add the following lines to `/etc/sysctl.d/60-hugepages.conf`:

    vm.nr_hugepages=128 

You WILL need to run `sudo sysctl --system` for these settings to take effect on your system (or reboot). In some cases (many threads, very large CPU, etc) you may need more than 128 (try 256 if there are still complaints from thread inits)

To increase the memlock (`ulimit -l`), add following lines to `/etc/security/limits.d/60-memlock.conf`:

    *    - memlock 262144 root - memlock 262144 

You WILL need to log out and log back in for these settings to take effect on your user (no need to reboot, just relogin in your session). Recheck after completing these steps to validate:

    $ /sbin/sysctl vm.nr_hugepages ; ulimit -l vm.nr_hugepages = 128 262144 

You can also do it Windows-style and simply run-as-root, but this is NOT recommended for security reasons. Also running as root does not properly get around the `ulimit -l` being large enough (and limits `*` does not apply to `root` either, it must be specified explicitly).


### 8. msvcp140.dll and vcruntime140.dll are not available
Download and install this [runtime package](https://go.microsoft.com/fwlink/?LinkId=746572) from Microsoft.

>***Warning***\*: Do NOT use "missing dll" sites - dll's are exe files with another name, and it is a fairly safe bet that any dll on a shady site like that will be trojaned. Please download offical runtimes from Microsoft above.\*

 

###9. Obtaining SeLockMemoryPrivilege failed.
For professional versions of Windows see [this article](https://msdn.microsoft.com/en-gb/library/ms190730.aspx). Make sure to reboot afterwards!

**For Windows 7/10 Home:**

1. Download and install [Windows Server 2003 Resource Kit Tools](https://www.microsoft.com/en-us/download/details.aspx?id=17657). Ignore any incompatibility warning during installation.
2. Open cmd or PowerShell as an administrator.
3. `Use ntrights -u %USERNAME% +r SeLockMemoryPrivilege`where `%USERNAME%` is the user that will be running the program.
4. Reboot.

Reference: [http://rybkaforum.net/cgi-bin/rybkaforum/topic\_show.pl?pid=259791#pid259791](http://rybkaforum.net/cgi-bin/rybkaforum/topic_show.pl?pid=259791#pid259791)

*Warning: Do not download ntrights.exe from any other site other than the offical Microsoft download page.*


### 10. Share rejected - Low diff share
Check if a coin that you are mining has changed algorithm in one of its forks and you use right hashing algorithm in pools.txt (parameter: `currency`).

 

### 11. VirtualAlloc failed
If you set up the user rights properly ([see issue #7](https://www.reddit.com/r/XmrStak/wiki/troubleshooting#wiki_7._memory_alloc_failed.3A_mmap_failed)), and your system has 4-8GB of RAM (and 50%+ is in use), there is a significant chance that there simply won't be a large enough chunk of contiguous memory because Windows is fairly bad at mitigating memory fragmentation.

If that happens, disable all auto-starting applications and run the miner after a reboot.

 
### 12. (Ubuntu compiling) - Nvidia insufficient driver
If you have this error after compiling xmr-stak in Ubuntu - make sure you have the latest drivers and not X.org.X Nouveau or v390. Install them manually or with [cuda package](https://www.reddit.com/r/XmrStak/wiki/guides/startup#wiki_2._ubuntu_18.10_setup_.2B_nvidia_.28compiling_from_source.29)

 

### 13. (Ubuntu compiling) - Could NOT find OpenCL (missing: OpenCL_LIBRARY OpenCL_INCLUDE_DIR) Cmake error at CmakeLists.txt
When [compiling in Ubuntu with Nvidia](https://www.reddit.com/r/XmrStak/wiki/guides/startup#wiki_2._ubuntu_18.10_setup_.2B_nvidia_.28compiling_from_source.29) devices, and running `cmake ..` command add additional param that disables OpenCL:  `cmake .. -DOpenCL_ENABLE=OFF` 

 

### 14. (Ubuntu compiling) - gcc v8 is not supported
Cuda 10 ships with gcc and g++ ver.8 which is not supported. Make sure you [set gcc and g++ to v6](https://www.reddit.com/r/XmrStak/wiki/guides/startup#wiki_2.2_compiling) before compiling. (step 2.2.6)




