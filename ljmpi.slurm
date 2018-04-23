#!/bin/sh 
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:14:00
#SBATCH --clusters=baobab
#SBATCH --output=slurm-%J.out
#SBATCH --exclusive
ml purge
ml GCC/5.4.0-2.26 icc/2016.3.210-GCC-5.4.0-2.26  OpenMPI/1.10.3 imkl/11.3.3.210
mpirun bin/dataset -p $1 -n 2000 -g 2.0 -f 200 -F 500 -d 2.0 -S 19937 -C $2
