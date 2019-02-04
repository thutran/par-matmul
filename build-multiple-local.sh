#!/bin/bash
for i in 8 16 32 64 128 192
do
	make BLOCK_SIZE=$i
done
