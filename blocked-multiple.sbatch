#!/bin/bash -l 
#SBATCH -n 1
#SBATCH -p RM
#SBATCH -t 00:10:00 
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err 

#echo $run " s_"$size
${SLURM_SUBMIT_DIR}/./benchmark-blocked-${size}

