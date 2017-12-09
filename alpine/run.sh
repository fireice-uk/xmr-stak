#!/bin/sh
# License: Unspecified
# Version: 0.2

source ./setup.sh

# setup config
setup_variables
[[ ! -f "config.txt" ]] && { setup_config; }
[[ ! -f "cpu.txt" ]] && { setup_cpu; }

# execute
exec xmr-stak
