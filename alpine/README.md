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

#### logs

``` bash
docker logs -f xmr
```
