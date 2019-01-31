#!/bin/bash
for s in 2 4 8 16 32 41 64 128
do
	name=block_size_$s
	mkdir $name
	
	for r in {1..20}
	do
		qsub -N "b-n-s_$s" -v size=$s,run=$r -o $name/$name-$r.out -e $name/$name-$r.err blocked-multiple.pbs
	done
done
