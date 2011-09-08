#!/bin/sh

OPTIONS="--i=100 --src=randomize --num-gpus=3 --quick"
SUFFIX="default.gtx480.3x"

mkdir -p eval/$SUFFIX

for i in audikw1.graph cage15.graph coPapersCiteseer.graph kkt_power.graph kron_g500-logn20.graph 
do
	echo ./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS  
	./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt
	sleep 10 
	echo ./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --mark-parents
	./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in europe.osm.graph hugebubbles-00020.graph
do
	echo ./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10
	./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 --mark-parents
	./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in nlpkkt160.graph
do
	echo ./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10
	./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 --mark-parents
	./bin/test_bfs_4.0_x86_64 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in wikipedia-20070206.mtx
do
	echo ./bin/test_bfs_4.0_x86_64 market ../../../graphs/$i $OPTIONS 
	./bin/test_bfs_4.0_x86_64 market ../../../graphs/$i $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_x86_64 market ../../../graphs/$i $OPTIONS --mark-parents
	./bin/test_bfs_4.0_x86_64 market ../../../graphs/$i $OPTIONS --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

echo /bin/test_bfs_4.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS 
./bin/test_bfs_4.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid2d.5000.$SUFFIX.txt	
	sleep 10 
echo /bin/test_bfs_4.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS --mark-parents 
./bin/test_bfs_4.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS --mark-parents > eval/$SUFFIX/grid2d.5000.$SUFFIX.parent.txt	
	sleep 10 

echo /bin/test_bfs_4.0_x86_64 grid3d 300 --queue-sizing=0.15 $OPTIONS 
./bin/test_bfs_4.0_x86_64 grid3d 300 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid3d.300.$SUFFIX.txt	
	sleep 10 
echo /bin/test_bfs_4.0_x86_64 grid3d 300 --queue-sizing=0.15 $OPTIONS --mark-parents 
./bin/test_bfs_4.0_x86_64 grid3d 300 --queue-sizing=0.15 $OPTIONS --mark-parents > eval/$SUFFIX/grid3d.300.$SUFFIX.parent.txt	
	sleep 10 

i=random.2Mv.128Me.gr
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS 
./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS --mark-parents 
./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
 
i=rmat.2Mv.128Me.gr
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS 
./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS --mark-parents 
./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i $OPTIONS --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt
	sleep 10 
