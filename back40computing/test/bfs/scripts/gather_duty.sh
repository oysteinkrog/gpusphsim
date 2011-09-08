#!/bin/bash


for j in serial warp scan scan-warp scan-warp-cta
do 

	for i in audikw1.graph cage15.graph coPapersCiteseer.graph europe.osm.graph hugebubbles-00020.graph kkt_power.graph kron_g500-logn20.graph
	do

		echo ./bin/microbench_bfs_4.0_i386.gather.$j metis ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --uneven

		./bin/microbench_bfs_4.0_i386.gather.$j metis ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --uneven | grep Duty > duty/$i.duty.$j.out

	done
	
	for i in nlpkkt160.graph
	do

		echo ./bin/microbench_bfs_4.0_i386.gather.$j metis ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --queue-sizing=0.5

		./bin/microbench_bfs_4.0_i386.gather.$j metis ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --queue-sizing=0.5 | grep Duty > duty/$i.duty.$j.out

	done
	
	for i in wikipedia-20070206.mtx
	do

		echo ./bin/microbench_bfs_4.0_i386.gather.$j market ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --uneven

		./bin/microbench_bfs_4.0_i386.gather.$j market ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --uneven | grep Duty > duty/$i.duty.$j.out

	done
	
	for i in rmat.2Mv.128Me.gr random.2Mv.128Me.gr
	do

		echo ./bin/microbench_bfs_4.0_x86_64.gather.$j dimacs ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --uneven

		./bin/microbench_bfs_4.0_x86_64.gather.$j dimacs ../../../graphs/$i --src=randomize --i=100 --quick --device=1 --instrumented --uneven | grep Duty > duty/$i.duty.$j.out

	done
	

	echo ./bin/microbench_bfs_4.0_i386.gather.$j grid2d 5000 --src=randomize --i=100 --quick --device=1 --instrumented --queue-sizing=0.5
	./bin/microbench_bfs_4.0_i386.$j grid2d 5000 --src=randomize --i=100 --quick --device=1 --instrumented --queue-sizing=0.5 | grep Duty > duty/grid2d5000.duty.$j.out
	
	echo ./bin/microbench_bfs_4.0_i386.gather.$j grid3d 300 --src=randomize --i=100 --quick --device=1 --instrumented --queue-sizing=0.5
	./bin/microbench_bfs_4.0_i386.$j grid3d 300 --src=randomize --i=100 --quick --device=1 --instrumented --queue-sizing=0.5 | grep Duty > duty/grid3d300.duty.$j.out

done
