#!/bin/bash 
for s in 2 4 8 16 32 64 128
do
	./benchmark-blocked-$s> block_size_$s.out
done
