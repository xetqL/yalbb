#!/bin/bash
PE=$1
PARTICLES=$2
NBIT=$3
CYCLE=$(( NBIT / 100 ))
CURRENTCALL=$4
PART=$5

while [ $CURRENTCALL -lt $NBIT ]; do
    sbatch -n $PE -p $5 -t 0-00:05:00 ljmpi.slurm $PE $PARTICLES $NBIT $CURRENTCALL
    let CURRENTCALL=$CURRENTCALL+100
done
