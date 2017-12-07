# CPU ONLY

## xmr-stak alpine

### Build

``` bash
docker build -t xmr-stak-alpine -f ./Dockerfile .
```

### Usage

#### run

required:
* -e WALLET_ADDRESS=
* IMAGE

``` bash
docker run --name xmr -d -e WALLET_ADDRESS= xmr-stak-alpine
```

#### config

All config settings are changeable by environment variables

* POOL_ADDRESS          - default: pool.supportxmr.com:333
* WALLET_ADDRESS        - required!
* POOL_PASSWORD         - default: x
* USE_TLS               - default: false
* TLS_FINGERPRINT       - optional
* USE_NICEHASH          - default: true
* POOL_WEIGHT           - default: 1
* CURRENCY              - default: monero
* CALL_TIMEOUT          - default: 10
* RETRY_TIME            - default: 30
* GIVEUP_LIMIT          - default: 0
* VERBOSE_LEVEL         - default: 4
* PRINT_MOTD            - default: true
* H_PRINT_TIME          - default: 60
* AES_OVERRIDE          - default: null
* USE_SLOW_MEMORY       - default: warn
* TLS_SECURE_ALGO       - default: true
* DAEMON_MODE           - default: false
* FLUSH_STDOUT          - default: true
* OUTPUT_FILE           - optional
* HTTPD_PORT            - optional
* HTTP_LOGIN            - optional
* HTTP_PASS             - optional
* PREFER_IPV4           - default: true
* CUSTOM_CPU            - default: false
* THREADS               - default: 1
* NO_PREFETCH           - default: true
* LOW_POWER_MODE        - default: false
* AFFINE_TO_CPU         - default: true

##### CPU

To setup custom CPU config, CUSTOM_CPU should be set to true

To affine to cpu; set AFFINE_TO_CPU true, CPU(s) list start from 0 and this is
taken into concideration when defining THREADS

e.g.:


``` bash
AFFINE_TO_CPU=true
# if THREADS=1
    { "low_power_mode" : false, "no_prefetch" : true, "affine_to_cpu" : 0 },
# or THREADS=3
    { "low_power_mode" : false, "no_prefetch" : true, "affine_to_cpu" : 0 },
    { "low_power_mode" : false, "no_prefetch" : true, "affine_to_cpu" : 1 },
    { "low_power_mode" : false, "no_prefetch" : true, "affine_to_cpu" : 2 },
```

#### logs

``` bash
docker logs -f xmr-stak-alpine
```
