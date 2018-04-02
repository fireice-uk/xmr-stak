#!/usr/bin/env bash


cmake . -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF
make

azure_script/create_cpu_config.sh > bin/cpu.txt
azure_script/create_pool_config.sh > bin/pools.txt
cp azure_script/config.txt bin/config.txt

