#!/usr/bin/env bash

if [ -z "$1" ] ||  [ $1 -eq 0 ]
    then
    NUMCORES=$(nproc)
    else
    NUMCORES=$1
 fi
 
echo '"cpu_threads_conf" :'
echo '['
 
for (( c=1; c<=$NUMCORES; c++ ))
do
    echo "{ \"low_power_mode\" : false, \"no_prefetch\" : true, \"affine_to_cpu\" : $c },"
done

echo '],'

 
 
