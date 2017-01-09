### XMR-Stak-CPU - Monero mining software

XMR-Stak is a universal Stratum pool miner. This is the CPU-mining version, but I'm planning to release Nvidia and AMD GPU versions too.


#### Usage on Windows 
1) Edit the config.txt file to enter your pool login and password. 
2) Double click the exe file. 

XMR-Stak should compile on any C++11 compliant compiler. Windows compiler is assumed to be MSVC 2015 CE. MSVC build environment is not vendored.
```
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

Windows binary release checksums

sha1sum xmr-stak-cpu.exe
4bc77aa89b046d0ccc62c79b6d9b6329248327a6  xmr-stak-cpu.exe

sha3sum xmr-stak-cpu.exe
c7c68a75b1ac322834b2708f94b2513c452bbac7ceed9a30b6b3b580  xmr-stak-cpu.exe

date
Tue  3 Jan 13:54:40 GMT 2017
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v2

iQEcBAEBCAAGBQJYa61hAAoJEPsk95p+1Bw05fcH/jwsMBCnS9NllprPJCm+L3PW
9mw2AsWKnPTjfNsV80LWFnl/Sl8VqPlrGiRVcqYtijqgpipGP7APunhB8uKf2pL0
TKKkMn0VX9fcBsBAQb9KYaT8zLuOqaY0rjXcHyHGv2mvRaa5EYeC7uK0pKBgPhOt
ccoRQjxqA1CEfQJICf0WEyaabUnSKfxCQ0dwADD0KMxD6EiJENt3m1+FSn0bA1j/
16l7caqmP4HfyFcoYptF5YRFnf+5V/8lwPMOxIDvfbhHKleX4i7yQDDLcvFynSig
1FVzcIG4c6Fv9FIYPqQEbnN+hU+odnpxbmipJhTEnSmj632pJ9R6Kr8wU45MBkA=
=K/fC
-----END PGP SIGNATURE-----
```

#### Usage on Linux
```
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

#### Example reports
```
HASHRATE REPORT
| ID | 2.5s |  60s |  15m | ID | 2.5s |  60s |  15m |
|  0 | 31.7 | 30.7 | 30.5 |  1 | 30.6 | 30.6 | 30.6 |
|  2 | 30.3 | 30.6 | 30.6 |  3 | 30.6 | 30.6 | 30.6 |
|  4 | 35.3 | 35.5 | 35.6 |  5 | 35.7 | 35.7 | 35.7 |
|  6 | 35.4 | 35.6 | 35.6 |  7 | 35.7 | 35.7 | 35.7 |
|  8 | 31.7 | 30.7 | 30.5 |  9 | 30.6 | 30.6 | 30.6 |
| 10 | 30.4 | 30.6 | 30.6 | 11 | 30.6 | 30.6 | 30.6 |
-----------------------------------------------------
Totals:   388.7 388.7 388.7 H/s
Highest:  388.7 H/s
```

```
RESULT REPORT
Difficulty       : 8192
Good results     : 5825 / 5826 (100.0 %)
Avg result time  : 10.3 sec
Pool-side hashes : 22683648

Top 10 best results found:
|  0 |         15407238 |  1 |         12699745 |
|  2 |         12194202 |  3 |          6999845 |
|  4 |          5533935 |  5 |          5315338 |
|  6 |          4700351 |  7 |          4500227 |
|  8 |          4023567 |  9 |          4021473 |

Error details:
| Count |                       Error text |           Last seen |
|     1 | [NETWORK ERROR]                  | 2017-01-02 21:29:15 |
```

```
CONNECTION REPORT
Connected since : 2017-01-02 21:29:40
Pool ping time  : 288 ms

Network error log:
| Date                |                                                       Error text |
| 2017-01-02 21:29:15 | CALL error: Timeout while waiting for a reply                    |
| 2017-01-02 21:29:30 | CONNECT error: GetAddrInfo: Name or service not known            |
```

#### Default dev donation
By default the miner will donate 1% of the hashpower (1 minute in 100 minutes) to my pool. If you want to change that, edit **donate-level.h** before you build the binaries.

If you want to donate directly to support further development, here is my wallet
* 4581HhZkQHgZrZjKeCfCJxZff9E3xCgHGF25zABZz7oR71TnbbgiS7sK9jveE6Dx6uMs2LwszDuvQJgRZQotdpHt1fTdDhk



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



