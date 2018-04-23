#!/bin/bash
PE=$1
PARTICLES=$2
NBIT=$3
CYCLE=$(( NBIT / 100 ))
if [ -z "$4" ]; then
    CURRENTCALL=0
else
    CURRENTCALL=$4
fi
while [ $CURRENTCALL -lt $NBIT ]; do
    sbatch -n $PE -p parallel -t 2-00:00:00 ljmpi.slurm $PE $PARTICLES $NBIT $CURRENTCALL
    let CURRENTCALL=$CURRENTCALL+100
done


