# HowTo Use xmr-stak

## Content Overview
* [Configuration](#configuration)
* [Usage on Windows](#usage-on-windows)
* [Usage on Linux](#usage-on-linux)
* [Command Line Options](#command-line-options)
* [HTML and JSON API report configuraton](#xx)

## Configurations

Before you started the miner the first time there are no config files available.
Config files will be created at the first start.
The number of files depends on the available backends.
`config.txt` contains the common miner settings.
`amd.txt`, `cpu.txt` and `nvidia.txt` contains miner backend specific settings and can be used for further tuning ([Tuning Guide](tuning.md)).


## Usage on Windows
1) Double click the `xmr-stak.exe` file
2) Fill in the pool url, username and password

## Usage on Linux
1) Open a terminal within the folder with the binary
2) Start the miner with `./xmr-stak`

## Command Line Options

The miner allow to overwrite some of the settings via command line options.

```
Usage: xmr-stak [OPTION]...

  -h, --help            show this help
  -v, --version         show version number
  -V, --version-long  show long version number
  -c, --config FILE     common miner configuration file
  --currency NAME       currency to mine: monero or aeon
  --noCPU               disable the CPU miner backend
  --cpu FILE            CPU backend miner config file
  --noAMD               disable the AMD miner backend
  --amd FILE            AMD backend miner config file

The Following options temporary overwrites the config file settings:
  -o, --url URL         pool url and port, e.g. pool.usxmrpool.com:3333
  -u, --user USERNAME   pool user name or wallet address
  -p, --pass PASSWD     pool password, in the most cases x or empty ""
```

## HTML and JSON API report configuraton

To configure the reports shown on the [README](README.md) side you need to edit the httpd_port variable. Then enable wifi on your phone and navigate to [miner ip address]:[httpd_port] in your phone browser. If you want to use the data in scripts, you can get the JSON version of the data at url [miner ip address]:[httpd_port]/api.json
