#!/usr/bin/env bash


cmake . -DCUDA_ENABLE=OFF -DOpenCL_ENABLE=OFF
make

azure_script/create_cpu_config.sh > bin/cpu.txt
