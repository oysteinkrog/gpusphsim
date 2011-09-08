#!/bin/sh

OPTIONS="--i=100 --src=randomize --num-gpus=4 --queue-sizing=0.95 --quick"

for j in 2 4 8 16
do 

	for i in 1 2 4 8
	do
	
		if [ -e ../../../graphs/random/random.${j}Mv.$((j*8))Me.gr.$((i-1)) ]
		then
		
			echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/random/random.${j}Mv.$((j*8))Me.gr --splice=$i $OPTIONS ... eval/random.4x.$((i*8))EF.${j}Mv.$((i*j*8))Me.txt
			./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/random/random.${j}Mv.$((j*8))Me.gr --splice=$i $OPTIONS > eval/random.4x.$((i*8))EF.${j}Mv.$((i*j*8))Me.txt

			echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/random/random.${j}Mv.$((j*8))Me.gr --splice=$i --mark-parents $OPTIONS ... eval/random.4x.$((i*8))EF.${j}Mv.$((i*j*8))Me.parents.txt
			./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/random/random.${j}Mv.$((j*8))Me.gr --splice=$i --mark-parents $OPTIONS > eval/random.4x.$((i*8))EF.${j}Mv.$((i*j*8))Me.parents.txt

		fi
	done
done


