#!/bin/bash 
for s in 8 16 32 64 128 192
do
	./benchmark-blocked-$s> block_size_$s.out
done
