#!/bin/bash

set -ex

xmr-stak \
    --benchmark 8 \
    --currency monero \
    --httpd 0 \
    --url pool.usxmrpool.com:3333 \
    --user x \
    --pass x \
    --rigid x

