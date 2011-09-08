#!/bin/bash

OPTIONS="--src=randomize --i=50 --quick --device=1"

for j in history warp label bitmask   
do 
	OUTDIR="cull/$j"
	
	mkdir -p $OUTDIR

	echo ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j grid2d 5000 $OPTIONS --queue-sizing=0.5
	     ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j grid2d 5000 $OPTIONS --queue-sizing=0.5 | grep Culling > $OUTDIR/grid2d5000.duty.$j.out
	
	echo ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j grid3d 300 $OPTIONS --queue-sizing=0.5
	     ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j grid3d 300 $OPTIONS --queue-sizing=0.5 | grep Culling > $OUTDIR/grid3d300.duty.$j.out

	for i in audikw1.graph cage15.graph coPapersCiteseer.graph europe.osm.graph hugebubbles-00020.graph kkt_power.graph kron_g500-logn20.graph
	do
		echo ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j metis ../../../graphs/$i $OPTIONS --uneven
		     ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j metis ../../../graphs/$i $OPTIONS --uneven | grep Culling > $OUTDIR/$i.duty.$j.out
	done
	
	for i in nlpkkt160.graph
	do
		echo ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j metis ../../../graphs/$i $OPTIONS --queue-sizing=0.5
		     ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j metis ../../../graphs/$i $OPTIONS --queue-sizing=0.5 | grep Culling > $OUTDIR/$i.duty.$j.out

	done
	
	for i in wikipedia-20070206.mtx
	do
		echo ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j market ../../../graphs/$i $OPTIONS --uneven
		     ./bin/lookup_cull/microbench_bfs_4.0_i386.cull.$j market ../../../graphs/$i $OPTIONS --uneven | grep Culling > $OUTDIR/$i.duty.$j.out

	done
	
	for i in rmat.2Mv.128Me.gr random.2Mv.128Me.gr
	do
		echo ./bin/lookup_cull/microbench_bfs_4.0_x86_64.cull.$j dimacs ../../../graphs/$i $OPTIONS --uneven
		     ./bin/lookup_cull/microbench_bfs_4.0_x86_64.cull.$j dimacs ../../../graphs/$i $OPTIONS --uneven | grep Culling > $OUTDIR/$i.duty.$j.out

	done
	
done
