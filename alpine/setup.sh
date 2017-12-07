#!/hint/sh

setup_variables() {
        # ${parameter:?}        - error out if parameter null or not set
        # ${parameter:-default} - set default value if parameter not set
        POOL_ADDRESS=${POOL_ADDRES:-pool.supportxmr.com:3333}
        WALLET_ADDRESS=${WALLET_ADDRESS:?}
        POOL_PASSWORD=${POOL_PASSWORD:-x}
        USE_TLS=${USE_TLS:-false}
        TLS_FINGERPRINT=${TLS_FINGERPRINT}
        USE_NICEHASH=${USE_NICEHASH:-true}
        POOL_WEIGHT=${POOL_WEIGHT:-1}
        CURRENCY=${CURRENCY:-monero}
        CALL_TIMEOUT=${CALL_TIMEOUT:-10}
        RETRY_TIME=${RETRY_TIME:-30}
        GIVEUP_LIMIT=${GIVEUP_LIMIT:-0}
        VERBOSE_LEVEL=${VERBOSE_LEVEL:-4}
        PRINT_MOTD=${PRINT_MOTD:-true}
        H_PRINT_TIME=${H_PRINT_TIME:-60}
        AES_OVERRIDE=${AES_OVERRIDE:-null}
        USE_SLOW_MEMORY=${USE_SLOW_MEMORY:-warn}
        TLS_SECURE_ALGO=${TLS_SECURE_ALGO:-true}
        DAEMON_MODE=${DAEMON_MODE:-false}
        FLUSH_STDOUT=${FLUSH_STDOUT:-true}
        OUTPUT_FILE=${OUTPUT_FILE}
        HTTPD_PORT=${HTTPD_PORT:-0}
        HTTP_LOGIN=${HTTP_LOGIN}
        HTTP_PASS=${HTTP_PASS}
        PREFER_IPV4=${PREFER_IPV4:-true}
}

setup_config() {
        local config=config.txt

        # create configuration
        (command touch "$config") && {
                # pool list
                printf "\"%s\" :\n"           "pool_list"
                printf "[{\n"
                # string
                printf "  \"%s\" : \"%s\",\n" "pool_address"    "$POOL_ADDRESS"
                printf "  \"%s\" : \"%s\",\n" "wallet_address"  "$WALLET_ADDRESS"
                printf "  \"%s\" : \"%s\",\n" "pool_password"   "$POOL_PASSWORD"
                printf "  \"%s\" : \"%s\",\n" "tls_fingerprint" "$TLS_FINGERPRINT"
                # boolean
                printf "  \"%s\" : %s,\n"     "use_nicehash"    "$USE_NICEHASH"
                printf "  \"%s\" : %s,\n"     "use_tls"         "$USE_TLS"
                # integer
                printf "  \"%s\" : %s\n"      "pool_weight"     "$POOL_WEIGHT"
                printf "},],\n"
                # string
                printf "\"%s\" : \"%s\",\n"   "currency"        "$CURRENCY"
                printf "\"%s\" : \"%s\",\n"   "use_slow_memory" "$USE_SLOW_MEMORY"
                printf "\"%s\" : \"%s\",\n"   "output_file"     "$OUTPUT_FILE"
                printf "\"%s\" : \"%s\",\n"   "http_login"      "$HTTP_LOGIN"
                printf "\"%s\" : \"%s\",\n"   "http_pass"       "$HTTP_PASS"
                # boolean
                printf "\"%s\" : %s,\n"       "aes_override"    "$AES_OVERRIDE"
                printf "\"%s\" : %s,\n"       "print_motd"      "$PRINT_MOTD"
                printf "\"%s\" : %s,\n"       "tls_secure_algo" "$TLS_SECURE_ALGO"
                printf "\"%s\" : %s,\n"       "daemon_mode"     "$DAEMON_MODE"
                printf "\"%s\" : %s,\n"       "flush_stdout"    "$FLUSH_STDOUT"
                printf "\"%s\" : %s,\n"       "prefer_ipv4"     "$PREFER_IPV4"
                # integer
                printf "\"%s\" : %s,\n"       "call_timeout"    "$CALL_TIMEOUT"
                printf "\"%s\" : %s,\n"       "retry_time"      "$RETRY_TIME"
                printf "\"%s\" : %s,\n"       "giveup_limit"    "$GIVEUP_LIMIT"
                printf "\"%s\" : %s,\n"       "verbose_level"   "$VERBOSE_LEVEL"
                printf "\"%s\" : %s,\n"       "h_print_time"    "$H_PRINT_TIME"
                printf "\"%s\" : %s,\n"       "httpd_port"      "$HTTPD_PORT"
        } >> "$config"
}

# vim: set ft=sh ts=8 sw=8 et:
