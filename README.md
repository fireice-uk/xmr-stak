### XMR-Stak-CPU - Monero mining software

XMR-Stak is a universal Stratum pool miner. This is the CPU-mining version; there is also an [AMD GPU version](https://github.com/fireice-uk/xmr-stak-amd) and an [NVIDA GPU version](https://github.com/fireice-uk/xmr-stak-nvidia)

#### HTML reports
<img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-hashrate.png" width="300"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-results.png" width="300"> <img src="https://gist.githubusercontent.com/fireice-uk/2da301131ac01695ff79539a27b81d68/raw/4c09cdeee86f94df2e9dd86b927e64aded6184f5/xmr-stak-cpu-connection.png" width="300">

#### Usage on Windows 
1) Edit the config.txt file to enter your pool login and password. 
2) Double click the exe file. 

XMR-Stak should compile on any C++11 compliant compiler. Windows compiler is assumed to be MSVC 2015 CE. MSVC build environment is not vendored.
```
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

sha1sum
d34a0ba0dd7b3b1f900a7e02772e197e974b4a73  libeay32.dll
2ee9966a0fc163da58408d91be36b84fa287c10b  ssleay32.dll
c406d5fbe4791b73271ab17a9b720fd15bcf5235  xmr-stak-cpu.exe
a6a63f70e19819ef9f65603267308ad9360c8256  xmr-stak-cpu-notls.exe

sha3sum
05003137a87313c81d6c348c9b96411c95d48dc22c35f36c39129747  libeay32.dll
133c065d9ef2c93396382e2ba5d8c3ca8c6a57c6beb0159cb9a4b6c5  ssleay32.dll
4d57aec0df1a9a3cb843b49aaebef3546fedd17a6adf7de77890909f  xmr-stak-cpu.exe
739e68fe4f76fe5d4ff314693f28baf0a166ca8e46dee0b0cd044370  xmr-stak-cpu-notls.exe

$ date
Wed 15 Mar 14:39:01 GMT 2017
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAEBCAAGBQJYyVJUAAoJEPsk95p+1Bw0sNIIAIPISmp5ut/zb7rlnbfxkrZf
PRuNF1g7+qMen1bUi9FTwoTeebz468kj+fbk8fKljYMyctOrgxbTkPFl7ag1k1UH
rjeiRMVWlw0FBEX+0n9D9Ots+whYmwYXXu8eXJFjyc9nzRxzJl//hLw7ASpwOBRx
C+HcFXJSEcUr5RcDxAR0jK0xHQiu82mp4QxxR1GpIRWfsbqQn+OOClhtedLFed/Y
rXivP2XH0SEAhZfa/JUMCcwkqldMOKotZqhZwOjLLGWkKGYUM04U9N58bGIg7tDj
lBoYbHZRpcWUfDO8o2y+ZQIs+yzMoJHHBBXB9fsHlwq62PTtzjsEVwB2aq9ABzk=
=S8cQ
-----END PGP SIGNATURE-----

```

#### Usage on Linux (Debian-based distros)
```
    sudo apt-get install libmicrohttpd-dev libssl-dev cmake build-essential
    cmake .
    make
```

GCC version 5.1 or higher is required for full C++11 support. CMake release compile scripts, as well as CodeBlocks build environment for debug builds is included.

To do a static build for a system without gcc 5.1+
```
    cmake -DCMAKE_BUILD_TYPE=STATIC
    make
```
Note - cmake caches variables, so if you want to do a dynamic build later you need to specify '-DCMAKE_BUILD_TYPE=RELEASE'

#### CPU mining performance 

Performance is nearly identical to the closed source paid miners. Here are some numbers:

* **I7-2600K** - 266 H/s
* **I7-6700** - 276 H/s (with a separate GPU miner)
* **Dual X5650** - 466 H/s (depends on NUMA)
* **Dual E5640** - 365 H/s (same as above)

#### Default dev donation
By default the miner will donate 1% of the hashpower (1 minute in 100 minutes) to my pool. If you want to change that, edit **donate-level.h** before you build the binaries.

If you want to donate directly to support further development, here is my wallet

```
4581HhZkQHgZrZjKeCfCJxZff9E3xCgHGF25zABZz7oR71TnbbgiS7sK9jveE6Dx6uMs2LwszDuvQJgRZQotdpHt1fTdDhk
```

#### PGP Key
```
-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: GnuPG v2

mQENBFhYUmUBCAC6493W5y1MMs38ApRbI11jWUqNdFm686XLkZWGDfYImzL6pEYk
RdWkyt9ziCyA6NUeWFQYniv/z10RxYKq8ulVVJaKb9qPGMU0ESfdxlFNJkU/pf28
sEVBagGvGw8uFxjQONnBJ7y7iNRWMN7qSRS636wN5ryTHNsmqI4ClXPHkXkDCDUX
QvhXZpG9RRM6jsE3jBGz/LJi3FyZLo/vB60OZBODJ2IA0wSR41RRiOq01OqDueva
9jPoAokNglJfn/CniQ+lqUEXj1vjAZ1D5Mn9fISzA/UPen5Z7Sipaa9aAtsDBOfP
K9iPKOsWa2uTafoyXgiwEVXCCeMMUjCGaoFBABEBAAG0ImZpcmVpY2VfdWsgPGZp
cmVpY2UueG1yQGdtYWlsLmNvbT6JATcEEwEIACEFAlhYUmUCGwMFCwkIBwIGFQgJ
CgsCBBYCAwECHgECF4AACgkQ+yT3mn7UHDTEcQf8CMhqaZ0IOBxeBnsq5HZr2X6z
E5bODp5cPs6ha1tjH3CWpk1AFeykNtXH7kPW9hcDt/e4UQtcHs+lu6YU59X7xLJQ
udOkpWdmooJMXRWS/zeeon4ivT9d69jNnwubh8EJOyw8xm/se6n48BcewfHekW/6
mVrbhLbF1dnuUGXzRN1WxsUZx3uJd2UvrkJhAtHtX92/qIVhT0+3PXV0bmpHURlK
YKhhm8dPLV9jPX8QVRHQXCOHSMqy/KoWEe6CnT0Isbkq3JtS3K4VBVeTX9gkySRc
IFxrNJdXsI9BxKv4O8yajP8DohpoGLMDKZKSO0yq0BRMgMh0cw6Lk22uyulGALkB
DQRYWFJlAQgAqikfViOmIccCZKVMZfNHjnigKtQqNrbJpYZCOImql4FqbZu9F7TD
9HIXA43SPcwziWlyazSy8Pa9nCpc6PuPPO1wxAaNIc5nt+w/x2EGGTIFGjRoubmP
3i5jZzOFYsvR2W3PgVa3/ujeYYJYo1oeVeuGmmJRejs0rp1mbvBSKw1Cq6C4cI0x
GTY1yXFGLIgdfYNMmiLsTy1Qwq8YStbFKeUYAMMG3128SAIaT3Eet911f5Jx4tC8
6kWUr6PX1rQ0LQJqyIsLq9U53XybUksRfJC9IEfgvgBxRBHSD8WfqEhHjhW1VsZG
dcYgr7A1PIneWsCEY+5VUnqTlt2HPaKweQARAQABiQEfBBgBCAAJBQJYWFJlAhsM
AAoJEPsk95p+1Bw0Pr8H/0vZ6U2zaih03jOHOvsrYxRfDXSmgudOp1VS45aHIREd
2nrJ+drleeFVyb14UQqO/6iX9GuDX2yBEHdCg2aljeP98AaMU//RiEtebE6CUWsL
HPVXHIkxwBCBe0YkJINHUQqLz/5f6qLsNUp1uTH2++zhdBWvg+gErTYbx8aFMFYH
0GoOtqE5rtlAh5MTvDZm+UcDwKJCxhrLaN3R3dDoyrDNRTgHQQuX5/opJBiUnVNK
d+vugnxzpMIJQP11yCZkz/KxV8zQ2QPMuZdAoh3znd/vGCJcp0rWphn4pqxA4vDp
c4hC0Yg9Dha1OoE5CJCqVL+ic4vAyB1urAwBlsd/wH8=
=B5I+
-----END PGP PUBLIC KEY BLOCK-----
```

### Common Issues

**SeLockMemoryPrivilege failed**

Please see [config.txt](config.txt) under section **LARGE PAGE SUPPORT**

For Windows 7 pro, or Windows 8 and above see [this article](https://msdn.microsoft.com/en-gb/library/ms190730.aspx)  (make sure to reboot afterwards!).

For Windows 7 Home :

1) Download and install [Windows Server 2003 Resource Kit Tools](https://www.microsoft.com/en-us/download/details.aspx?id=17657).  Ignore incompatiablity warning during installation.

2) In cmd or power shell: `ntrights -u %USERNAME% +r SeLockMemoryPrivilege`  (where %USERNAME% is the user that will be running the program.  This command needs to be run as admin)

3) Reboot.

Reference: http://rybkaforum.net/cgi-bin/rybkaforum/topic_show.pl?pid=259791#pid259791

*Warning: do not download ntrights.exe from any other site other then the offical Microsoft download page.*

**VirtualAlloc failed**

If you set up the user rights properly (see above), and your system has 4-8GB of RAM (50%+ use), there is a significant chance that there simply won't be a large enough chunk of contiguous memory because Windows is fairly bad at mitigating memory fragmentation.

If that happens, disable all auto-staring applications and run the miner after a reboot.

**msvcp140.dll and vcruntime140.dll not available errors**

Download and install this [runtime package](https://www.microsoft.com/en-us/download/details.aspx?id=48145) from Microsoft.  *Warning: Do NOT use "missing dll" sites - dll's are exe files with another name, and it is a fairly safe bet that any dll on a shady site like that will be trojaned.  Please download offical runtimes from Microsoft above.*


**Error: MEMORY ALLOC FAILED: mmap failed**

From [config.txt](config.txt):

On Linux you will need to configure large page support `sudo sysctl -w vm.nr_hugepages=128` and increase your
ulimit -l. To do do this you need to add following lines to /etc/security/limits.conf:

    * soft memlock 262144
    * hard memlock 262144
    
Save file.  You WILL need to log out and log back in for these settings to take affect on your user (no need to reboot, just relogin in your session).
   
You can also do it Windows-style and simply run-as-root, but this is NOT recommended for security reasons.

**Illegal instruction (core dumped)**

This typically means you are trying to run it on a CPU that does not have [AES](https://en.wikipedia.org/wiki/AES_instruction_set).  This only happens on older version of miner, new version gives better error message (but still wont' work since your CPU doesn't support the required instructions).



