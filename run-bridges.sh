#!/bin/bash
for i in 2 4 8 16 32 41 64 128
do
	name=block_size_$i
	mkdir $name
	workdir=$PWD/$name/
	for r in {1..20} 
	do
		sbatch --export=size=$i,run=$r --job-name=$name --chdir=$workdir blocked-multiple.sbatch	
	done
done
