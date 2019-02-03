#!/bin/bash
for i in 16 32
do
	make BLOCK_SIZE=$i
done
