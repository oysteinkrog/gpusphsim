#!/bin/bash

mkdir -p eval

OPTIONS="--i=5 --n=32000000 --device=0 --verify"
OUTPUT="eval/GTX580.txt"
ARCH=200

echo 1B
./bin/tune_copy_4.0_i386_sm${ARCH}_u1B $OPTIONS > $OUTPUT
echo 2B
./bin/tune_copy_4.0_i386_sm${ARCH}_u2B $OPTIONS >> $OUTPUT
echo 4B
./bin/tune_copy_4.0_i386_sm${ARCH}_u4B $OPTIONS >> $OUTPUT
echo 8B
./bin/tune_copy_4.0_i386_sm${ARCH}_u8B $OPTIONS >> $OUTPUT
