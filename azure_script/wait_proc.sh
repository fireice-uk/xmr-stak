#!/usr/bin/env bash


if [ -z "$1" ]; then
    exit
fi

while [ -e /proc/$1 ]; do 

sleep 1; 

done
