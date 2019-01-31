#!/bin/bash
for i in 2 4 8 16 32 41 64 128
do
	make BLOCK_SIZE=$i
done
