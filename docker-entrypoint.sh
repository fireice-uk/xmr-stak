#!/usr/bin/env bash
set -e

# this if will check if the first argument is a flag
# but only works if all arguments require a hyphenated flag
# -v; -SL; -f arg; etc will work, but not arg1 arg2
if [ "${1:0:1}" = '-' ]; then
    set -- xmr-stak "$@"
fi

# automate setup of config
# TODO: I'm sure someone better at awk than me could do this in one command
# TODO: this would be easier if they used actual JSON or an actual config language like YAML
DEFAULT_CONF=/usr/local/src/xmr-stak/config.txt
USER_CONF=$HOME/config.txt

if [ ! -e "$USER_CONF" ]; then

    [ -z "$MAX_CPUS" ] && MAX_CPUS=99

    CPU_THREADS=$(xmr-stak "$DEFAULT_CONF" | awk '/BEGIN/{f=1;next} /END/{f=0} f' | head -n-2 | awk 'NR==1 {next} NR>3+'$MAX_CPUS' {next} {print} END {print "],"}' | tr '\n' "\\n")

    [ -z "$CPU_THREADS" ] && exit 99

    awk \
        -v c="$CPU_THREADS" \
        '{sub(/"cpu_threads_conf" : null,/,c)}1' \
        "$DEFAULT_CONF" >"$USER_CONF"

    [ -n "$VERBOSE_LEVEL" ] && \
        sed -i "s/^\"verbose_level\".*/\"verbose_level\" : $VERBOSE_LEVEL,/" "$USER_CONF"

    [ -n "$USE_SLOW_MEMORY" ] && \
        sed -i "s/^\"use_slow_memory\".*/\"use_slow_memory\" : \"$USE_SLOW_MEMORY\",/" "$USER_CONF"

    [ -n "$USE_TLS" ] && \
        sed -i "s/^\"use_tls\".*/\"use_tls\" : $USE_TLS,/" "$USER_CONF"

    [ -n "$POOL" ] && \
        sed -i "s~^\"pool_address\".*~\"pool_address\" : \"$POOL\",~" "$USER_CONF"

    [ -n "$LOGIN" ] && \
         sed -i "s/^\"wallet_address\".*/\"wallet_address\" : \"$LOGIN\",/" "$USER_CONF"

    [ -n "$PASS" ] && \
        sed -i "s/^\"pool_password\".*/\"pool_password\" : \"$PASS\",/" "$USER_CONF"

    # TODO: more config overrides

    # default to always using the http port
    [ -z "$HTTPD_PORT" ] && HTTPD_PORT=8000
    sed -i "s/^\"httpd_port\".*/\"httpd_port\" : $HTTPD_PORT,/" "$USER_CONF"
fi
cat "$USER_CONF"

# check for the expected command
if [ "$1" = "xmr-stak" ]; then
    # TODO: send logs somewhere with rotation
    exec "$@" "$USER_CONF"
fi

# otherwise, don't get in their way
exec "$@"
