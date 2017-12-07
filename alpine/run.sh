#!/bin/sh
# License: Unspecified
# Version: 0.1

source ./setup.sh

# setup config
setup_variables
setup_config

# execute
exec xmr-stak
