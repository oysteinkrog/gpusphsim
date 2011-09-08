#!/bin/sh

OPTIONS="--i=100 --src=randomize --device=2 --quick"
SUFFIX="default.gtx480.1x"

mkdir -p eval/$SUFFIX

for i in coPapersCiteseer.graph kkt_power.graph 
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS   
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt
	sleep 10 
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=1.1 --mark-parents 
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=1.1 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in cage15.graph
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.50
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.50 > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.50 --mark-parents
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.50 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in audikw1.graph kron_g500-logn20.graph 
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=1.15
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=1.15 > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --uneven --queue-sizing=1.15 --mark-parents
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --uneven --queue-sizing=1.15 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in europe.osm.graph hugebubbles-00020.graph
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 --mark-parents
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in nlpkkt160.graph
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --queue-sizing=0.10 > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --uneven --queue-sizing=0.10 --mark-parents
	     ./bin/test_bfs_4.0_i386 metis ../../../graphs/$i $OPTIONS --uneven --queue-sizing=0.10 --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

for i in wikipedia-20070206.mtx
do
	echo ./bin/test_bfs_4.0_i386 market ../../../graphs/$i $OPTIONS 
	     ./bin/test_bfs_4.0_i386 market ../../../graphs/$i $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt 
	sleep 10 
	echo ./bin/test_bfs_4.0_i386 market ../../../graphs/$i $OPTIONS --uneven --mark-parents
	     ./bin/test_bfs_4.0_i386 market ../../../graphs/$i $OPTIONS --uneven --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
	sleep 10 
done

echo ./bin/test_bfs_4.0_i386 grid2d 5000 --queue-sizing=0.15 $OPTIONS 
     ./bin/test_bfs_4.0_i386 grid2d 5000 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid2d.5000.$SUFFIX.txt	
sleep 10 
echo ./bin/test_bfs_4.0_i386 grid2d 5000 --queue-sizing=0.15 $OPTIONS --mark-parents 
     ./bin/test_bfs_4.0_i386 grid2d 5000 --queue-sizing=0.15 $OPTIONS --mark-parents > eval/$SUFFIX/grid2d.5000.$SUFFIX.parent.txt	
sleep 10 

echo ./bin/test_bfs_4.0_i386 grid3d 300 --queue-sizing=0.15 $OPTIONS 
     ./bin/test_bfs_4.0_i386 grid3d 300 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid3d.300.$SUFFIX.txt	
sleep 10 
echo ./bin/test_bfs_4.0_i386 grid3d 300 --queue-sizing=0.15 $OPTIONS --mark-parents 
./bin/test_bfs_4.0_i386 grid3d 300 --queue-sizing=0.15 $OPTIONS --mark-parents > eval/$SUFFIX/grid3d.300.$SUFFIX.parent.txt	
sleep 10 

i=random.2Mv.128Me.gr
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS 
     ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt 
sleep 10 
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS --mark-parents 
     ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt 
sleep 10 
 
i=rmat.2Mv.128Me.gr
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS 
     ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS > eval/$SUFFIX/$i.$SUFFIX.txt 
sleep 10 
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS --mark-parents 
     ./bin/test_bfs_4.0_x86_64 dimacs ../../../graphs/$i --uneven $OPTIONS --mark-parents > eval/$SUFFIX/$i.$SUFFIX.parent.txt
sleep 10 
