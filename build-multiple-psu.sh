#!/bin/bash
for i in 2 4 8 16 32 64 128 256 512
do
	make BLOCK_SIZE=$i CC=gcc -f Makefile-psu.make
done
