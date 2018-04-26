#!/bin/bash
PE=$1
PARTICLES=$2
NBIT=$3
CYCLE=$(( NBIT / 100 ))
NAME=$4
if [ -z "$5" ]; then
    CURRENTCALL=0
else
    CURRENTCALL=$4
fi

if [ -z "$6" ]; then
    DEST="BAOBAB"
else
    DEST="LOCAL"
fi

while [ $CURRENTCALL -lt $NBIT ]; do
    if [ "$DEST" == "LOCAL" ]; then
        mpirun -np $1 bin/dataset -p $1 -n $2 -g 2.0 -f $3 -F 1 -d 2.0 -S 19937 -C $CURRENTCALL
    else
        sbatch -n $PE -p parallel -t 2-00:00:00 -J $NAME ljmpi.slurm $PE $PARTICLES $NBIT $CURRENTCALL
    fi
    let CURRENTCALL=$CURRENTCALL+100
done