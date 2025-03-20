#! /bin/bash

lim=100
while :
do
    a=`nvidia-smi --query-gpu=memory.used --format=csv | cut -f 1 -d ' ' | tail -n 8`
    g0=`echo $a |cut -f 2 -d ' '`
    g1=`echo $a |cut -f 3 -d ' '`
    g2=`echo $a |cut -f 4 -d ' '`
    g3=`echo $a |cut -f 5 -d ' '`
    g4=`echo $a |cut -f 6 -d ' '`
    g5=`echo $a |cut -f 7 -d ' '`
    if [ $g0 -lt $lim ] && [ $g1 -lt $lim ] && [ $g2 -lt $lim ] && [ $g3 -lt $lim ] &&  [ $g4 -lt $lim ] && [ $g5 -lt $lim ] ; then
        bash ./llm_multi_gen_30b.sh

        exit
    fi
    sleep 1h
done

# sleep 8h
