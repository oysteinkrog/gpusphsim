#!/bin/sh

OPTIONS="--i=100 --src=randomize --quick --device=1"

DIR="eval/random.c2050.1x"
mkdir -p $DIR

# vertices
for j in 1 2 4 8 16 32
do 
	#edge factor
	for i in 8 16 32 64 128 256
	do
		if [ $((j*i)) -lt 256 ]
		then
		
			if [ $((j*i)) -lt 128 ]
			then
				echo 	./bin/test_bfs_4.0_i386 random $((j*1000000)) $((j*i*1000000)) $OPTIONS . $DIR/random.1x.$((i))EF.${j}Mv.$((i*j))Me.txt
						./bin/test_bfs_4.0_i386 random $((j*1000000)) $((j*i*1000000)) $OPTIONS > $DIR/random.1x.$((i))EF.${j}Mv.$((i*j))Me.txt
			else 
				echo 	./bin/test_bfs_4.0_x86_64 random $((j*1000000)) $((j*i*1000000)) $OPTIONS . $DIR/random.1x.$((i))EF.${j}Mv.$((i*j))Me.txt
						./bin/test_bfs_4.0_x86_64 random $((j*1000000)) $((j*i*1000000)) $OPTIONS > $DIR/random.1x.$((i))EF.${j}Mv.$((i*j))Me.txt
			fi

		elif [ $((j*i)) -eq 256 ]
		then
			echo 	./bin/test_bfs_4.0_x86_64 random $((j*1000000)) $((j*i*1000000)) --uneven $OPTIONS . $DIR/random.1x.$((i))EF.${j}Mv.$((i*j))Me.txt
					./bin/test_bfs_4.0_x86_64 random $((j*1000000)) $((j*i*1000000)) --uneven $OPTIONS > $DIR/random.1x.$((i))EF.${j}Mv.$((i*j))Me.txt
		fi
		
	done
done


