#!/usr/bin/env bash
 
 #Variables
 #pool_address1, pool_address2 
 #wallet1, wallet2
 #pool_pass1, #pool_pass2
 #nicehash1, nicehash2  -> true, false
 #tls1, tls2 -> true, false
 #tls_id1, tls_id2
 #pool_weight1, pool_weight2: 1..100
 

[ -z "$wallet1" ] &&  exit 100
[ -z "$pool_address1" ] &&  exit 101
[ -z "$pool_pass1" ] &&  pool_pass1=x
[ -z "$nicehash1" ] &&  nicehash1=false
[ -z "$tls1" ] &&  tls1=false
[ -z "$pool_weight1" ] &&  pool_weight1=100

 echo '
"pool_list" :
[
'
echo "{\"pool_address\" : \"$pool_address1\", \"wallet_address\" : \"$wallet1\", \"pool_password\" : \"$pool_pass1\", \"use_nicehash\" : $nicehash1, \"use_tls\" : $tls1, \"tls_fingerprint\" : \"$tls_id1\", \"pool_weight\" : $pool_weight1 },"

if [ -n "$wallet2" ]
then 

    [ -z "$pool_address2" ] &&  exit 201
    [ -z "$pool_pass2" ] &&  pool_pass2=x
    [ -z "$nicehash2" ] &&  nicehash2=false
    [ -z "$tls2" ] &&  tls2=false
    [ -z "$pool_weight2" ] &&  pool_weight2=1

    echo "{\"pool_address\" : \"$pool_address2\", \"wallet_address\" : \"$wallet2\", \"pool_password\" : \"$pool_pass2\", \"use_nicehash\" : $nicehash2, \"use_tls\" : $tls2, \"tls_fingerprint\" : \"$tls_id2\", \"pool_weight\" : $pool_weight2 },"
fi
	

echo '	
],
"currency" : "monero",
"call_timeout" : 10,
"retry_time" : 30,
"giveup_limit" : 0,
"verbose_level" : 3,
"print_motd" : true,
"h_print_time" : 60,
"aes_override" : null,
"use_slow_memory" : "warn",
"tls_secure_algo" : true,
"daemon_mode" : false,
"flush_stdout" : false,
"output_file" : "",
"httpd_port" : 0,
"http_login" : "",
"http_pass" : "",
"prefer_ipv4" : true,
'
