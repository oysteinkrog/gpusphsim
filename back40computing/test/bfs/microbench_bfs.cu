/******************************************************************************
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for BFS graph traversal.
 *
 * Useful for demonstrating how to integrate BFS traversal into your 
 * application. 
 ******************************************************************************/

#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <b40c_test_util.h>

// Graph construction utils
#include <b40c/graph/builder/dimacs.cuh>
#include <b40c/graph/builder/grid2d.cuh>
#include <b40c/graph/builder/grid3d.cuh>
#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/metis.cuh>
#include <b40c/graph/builder/random.cuh>
#include <b40c/graph/builder/rr.cuh>

// BFS includes
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/microbench/enactor_gather_lookup.cuh>

using namespace b40c;
using namespace graph;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

//#define __B40C_ERROR_CHECKING__		 

bool g_verbose;
bool g_verbose2;
bool g_undirected;
bool g_quick;			// Whether or not to perform CPU traversal as reference
bool g_uneven;


/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_bfs <graph type> <graph type args> [--device=<device index>] "
			"[--v] [--instrumented] [--i=<num-iterations>] [--undirected]"
			"[--src=< <source idx> | randomize >] [--queue-size=<queue size>\n"
			"[--mark-parents]\n"
			"\n"
			"graph types and args:\n"
			"\tgrid2d <width>\n"
			"\t\t2D square grid lattice with width <width>.  Interior vertices \n"
			"\t\thave 4 neighbors and 1 self-loop.  Default source vertex is the grid-center.\n"
			"\tgrid3d <side-length>\n"
			"\t\t3D square grid lattice with width <width>.  Interior vertices \n"
			"\t\thave 6 neighbors and 1 self-loop.  Default source vertex is the grid-center.\n"
			"\tdimacs [<file>]\n"
			"\t\tReads a DIMACS-formatted graph of directed edges from stdin (or \n"
			"\t\tfrom the optionally-specified file).  Default source vertex is random.\n" 
			"\tmetis [<file>]\n"
			"\t\tReads a METIS-formatted graph of directed edges from stdin (or \n"
			"\t\tfrom the optionally-specified file).  Default source vertex is random.\n" 
			"\tmarket [<file>]\n"
			"\t\tReads a Matrix-Market coordinate-formatted graph of directed edges from stdin (or \n"
			"\t\tfrom the optionally-specified file).  Default source vertex is random.\n"
			"\trandom <n> <m>\n"			
			"\t\tA random graph generator that adds <m> edges to <n> nodes by randomly \n"
			"\t\tchoosing a pair of nodes for each edge.  There are possibilities of \n"
			"\t\tloops and multiple edges between pairs of nodes. Default source vertex \n"
			"\t\tis random.\n"
			"\trr <n> <d>\n"			
			"\t\tA random graph generator that adds <d> randomly-chosen edges to each\n"
			"\t\tof <n> nodes.  There are possibilities of loops and multiple edges\n"
			"\t\tbetween pairs of nodes. Default source vertex is random.\n"
			"\n"
			"--v\tVerbose launch and statistical output is displayed to the console.\n"
			"\n"
			"--v2\tSame as --v, but also displays the input graph to the console.\n"
			"\n"
			"--instrumented\tKernels keep track of queue-search_depth, redundant work (i.e., the \n"
			"\t\toverhead of duplicates in the frontier), and average barrier duty (a \n"
			"\t\trelative indicator of load imbalance.)\n"
			"\n"
			"--i\tPerforms <num-iterations> test-iterations of BFS traversals.\n"
			"\t\tDefault = 1\n"
			"\n"
			"--src\tBegins BFS from the vertex <source idx>. Default is specific to \n"
			"\t\tgraph-type.  If alternatively specified as \"randomize\", each \n"
			"\t\ttest-iteration will begin with a newly-chosen random source vertex.\n"
			"\n"
			"--queue-sizing\tAllocates a frontier queue sized at (graph-edges * <queue-sizing>).  Default\n"
			"\t\tis 1.15.\n"
			"\n"
			"--mark-parents\tParent vertices are marked instead of source distances, i.e., it\n"
			"\t\tcreates an ancestor tree rooted at the source vertex.\n"
			"\n"
			"--stream-from-host\tKeeps the graph data (column indices, row offsets) on the host,\n"
			"\t\tusing zero-copy access to traverse it.\n"
			"\n"
			"--num-gpus\tNumber of GPUs to use\n"
			"\n"
			"--undirected\tEdges are undirected.  Reverse edges are added to DIMACS and\n"
			"\t\trandom graphs, effectively doubling the CSR graph representation size.\n"
			"\t\tGrid2d/grid3d graphs are undirected regardless of this flag, and rr \n"
			"\t\tgraphs are directed regardless of this flag.\n"
			"\n");
}

/**
 * Displays the BFS result (i.e., distance from source)
 */
template<typename VertexId, typename SizeT>
void DisplaySolution(VertexId* source_path, SizeT nodes)
{
	printf("[");
	for (VertexId i = 0; i < nodes; i++) {
		PrintValue(i);
		printf(":");
		PrintValue(source_path[i]);
		printf(", ");
	}
	printf("]\n");
}


/******************************************************************************
 * Performance/Evaluation Statistics
 ******************************************************************************/

struct Statistic 
{
	double mean;
	double m2;
	size_t count;
	
	Statistic() : mean(0.0), m2(0.0), count(0) {}
	
	/**
	 * Updates running statistic, returning bias-corrected sample variance.
	 * Online method as per Knuth.
	 */
	double Update(double sample)
	{
		count++;
		double delta = sample - mean;
		mean = mean + (delta / count);
		m2 = m2 + (delta * (sample - mean));
		return m2 / (count - 1);					// bias-corrected 
	}
	
};

struct Stats {
	char *name;
	Statistic rate;
	Statistic search_depth;
	Statistic redundant_work;
	Statistic expand_duty;
	Statistic compact_duty;
	
	Stats() : name(NULL), rate(), search_depth(), redundant_work(), expand_duty(), compact_duty() {}
	Stats(char *name) : name(name), rate(), search_depth(), redundant_work(), expand_duty(), compact_duty() {}
};


template <typename SizeT>
struct HistogramLevel
{
	SizeT		discovered;
	SizeT		expanded;
	SizeT		unique_expanded;

	HistogramLevel() : discovered(0), expanded(0), unique_expanded(0) {}
};


/**
 * Displays a histogram of search behavior by level depth, i.e., expanded,
 * unique, and newly-discovered nodes at each level
 */
template <
	typename VertexId,
	typename Value,
	typename SizeT>
void Histogram(
	VertexId 								src,
	VertexId 								*reference_source_dist,					// reference answer
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,	// reference host graph
	VertexId								search_depth)
{
	std::vector<HistogramLevel<SizeT> > histogram(search_depth + 1);
	std::vector<std::vector<VertexId> > frontier(search_depth + 1);

	// Establish basics
	histogram[0].expanded = 1;
	histogram[0].unique_expanded = 1;

	for (VertexId vertex = 0; vertex < csr_graph.nodes; vertex++) {

		VertexId distance = reference_source_dist[vertex];
		if (distance >= 0) {

			SizeT row_offset 	= csr_graph.row_offsets[vertex];
			SizeT row_oob 		= csr_graph.row_offsets[vertex + 1];
			SizeT neighbors 	= row_oob - row_offset;

			histogram[distance].discovered++;
			histogram[distance + 1].expanded += neighbors;
		}
	}

	// Allocate frontiers
	for (VertexId distance = 0; distance < search_depth; distance++) {
		frontier[distance].reserve(histogram[distance].expanded);
	}

	// Construct frontiers
	for (VertexId vertex = 0; vertex < csr_graph.nodes; vertex++) {

		VertexId distance = reference_source_dist[vertex];
		if (distance >= 0) {

			SizeT row_offset 	= csr_graph.row_offsets[vertex];
			SizeT row_oob 		= csr_graph.row_offsets[vertex + 1];

			frontier[distance].insert(
				frontier[distance].end(),
				csr_graph.column_indices + row_offset,
				csr_graph.column_indices + row_oob);
		}
	}

	printf("Work Histogram:\n");
	printf("Depth, Expanded, Unique-Expanded, Discovered\n");
	for (VertexId distance = 0; distance < search_depth; distance++) {

		// Sort
		std::sort(
			frontier[distance].begin(),
			frontier[distance].end());

		// Count unique elements
		histogram[distance + 1].unique_expanded =
			std::unique(frontier[distance].begin(), frontier[distance].end()) -
			frontier[distance].begin();

		printf("%d, %d, %d, %d\n",
			distance,
			histogram[distance].expanded,
			histogram[distance].unique_expanded,
			histogram[distance].discovered);
	}
	printf("\n\n");
}


/**
 * Displays timing and correctness statistics 
 */
template <
	bool MARK_PARENTS,
	typename VertexId,
	typename Value,
	typename SizeT>
void DisplayStats(
	Stats 									&stats,
	VertexId 								src,
	VertexId 								*h_source_path,							// computed answer
	VertexId 								*reference_source_dist,					// reference answer
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,	// reference host graph
	double 									elapsed,
	VertexId								search_depth,
	long long 								total_queued,
	double 									expand_duty,
	double 									compact_duty)
{
	// Compute nodes and edges visited
	SizeT edges_visited = 0;
	SizeT nodes_visited = 0;
	for (VertexId i = 0; i < csr_graph.nodes; i++) {
		if (h_source_path[i] > -1) {
			nodes_visited++;
			edges_visited += csr_graph.row_offsets[i + 1] - csr_graph.row_offsets[i];
		}
	}
	
	double redundant_work = 0.0;
	if (total_queued > 0)  {
		redundant_work = ((double) total_queued - edges_visited) / edges_visited;		// measure duplicate edges put through queue
	}
	redundant_work *= 100;

	// Display test name
	printf("[%s] finished. ", stats.name);

	// Display correctness
	if (reference_source_dist != NULL) {
		printf("Validity: ");
		fflush(stdout);
		if (!MARK_PARENTS) {

			// Simply compare with the reference source-distance
			CompareResults(h_source_path, reference_source_dist, csr_graph.nodes, true);

		} else {

			// Verify plausibility of parent markings
			bool correct = true;

			for (VertexId node = 0; node < csr_graph.nodes; node++) {
				VertexId parent = h_source_path[node];

				// Check that parentless nodes have zero or unvisited source distance
				VertexId node_dist = reference_source_dist[node];
				if (parent < 0) {
					if (reference_source_dist[node] > 0) {
						printf("INCORRECT: parentless node %lld (parent %lld) has positive distance distance %lld",
							(long long) node, (long long) parent, (long long) node_dist);
						correct = false;
						break;
					}
					continue;
				}

				// Check that parent has iteration one less than node
				VertexId parent_dist = reference_source_dist[parent];
				if (parent_dist + 1 != node_dist) {
					printf("INCORRECT: parent %lld has distance %lld, node %lld has distance %lld",
						(long long) parent, (long long) parent_dist, (long long) node, (long long) node_dist);
					correct = false;
					break;
				}

				// Check that parent is in fact a parent
				bool found = false;
				for (SizeT neighbor_offset = csr_graph.row_offsets[parent];
					neighbor_offset < csr_graph.row_offsets[parent + 1];
					neighbor_offset++)
				{
					if (csr_graph.column_indices[neighbor_offset] == node) {
						found = true;
						break;
					}
				}
				if (!found) {
					printf("INCORRECT: %lld is not a neighbor of %lld",
						(long long) parent, (long long) node);
					correct = false;
					break;
				}
			}

			if (correct) {
				printf("CORRECT");
			}

		}
	}
	printf("\n");

	// Display statistics
	if (nodes_visited < 5) {
		printf("Fewer than 5 vertices visited.\n");

	} else {
		
		// Display the specific sample statistics
		double m_teps = (double) edges_visited / (elapsed * 1000.0); 
		printf("\telapsed: %.3f ms, rate: %.3f MiEdges/s", elapsed, m_teps);
		if (search_depth != 0) printf(", search_depth: %lld", (long long) search_depth);
		if (expand_duty != 0) {
			printf("\n\texpand cta duty: %.2f%%", expand_duty * 100);
		}
		if (compact_duty != 0) {
			printf("\n\tcompact cta duty: %.2f%%", compact_duty * 100);
		}
		printf("\n\tsrc: %lld, nodes visited: %lld, edges visited: %lld",
			(long long) src, (long long) nodes_visited, (long long) edges_visited);
		if (total_queued > 0) {
			printf(", total queued: %lld", total_queued);
		}
		if (redundant_work > 0) {
			printf(", redundant work: %.2f%%", redundant_work);
		}
		printf("\n");

		// Display the aggregate sample statistics
		printf("\tSummary after %lld test iterations (bias-corrected):\n", (long long) stats.rate.count + 1);

		double search_depth_stddev = sqrt(stats.search_depth.Update((double) search_depth));
		if (search_depth > 0) printf(			"\t\t[Search depth]:           u: %.1f, s: %.1f, cv: %.4f\n",
			stats.search_depth.mean, search_depth_stddev, search_depth_stddev / stats.search_depth.mean);

		double redundant_work_stddev = sqrt(stats.redundant_work.Update(redundant_work));
		if (redundant_work > 0) printf(	"\t\t[redundant work %%]: u: %.2f, s: %.2f, cv: %.4f\n",
			stats.redundant_work.mean, redundant_work_stddev, redundant_work_stddev / stats.redundant_work.mean);

		double expand_duty_stddev = sqrt(stats.expand_duty.Update(expand_duty * 100));
		if (expand_duty > 0) printf(	"\t\t[Expand Duty %%]:        u: %.2f, s: %.2f, cv: %.4f\n",
			stats.expand_duty.mean, expand_duty_stddev, expand_duty_stddev / stats.expand_duty.mean);

		double compact_duty_stddev = sqrt(stats.compact_duty.Update(compact_duty * 100));
		if (compact_duty > 0) printf(	"\t\t[Compact Duty %%]:        u: %.2f, s: %.2f, cv: %.4f\n",
			stats.compact_duty.mean, compact_duty_stddev, compact_duty_stddev / stats.compact_duty.mean);

		double rate_stddev = sqrt(stats.rate.Update(m_teps));
		printf(								"\t\t[Rate MiEdges/s]:   u: %.3f, s: %.3f, cv: %.4f\n", 
			stats.rate.mean, rate_stddev, rate_stddev / stats.rate.mean);
	}
	
	fflush(stdout);

}
		

/******************************************************************************
 * BFS Testing Routines
 ******************************************************************************/

template <
	bool INSTRUMENT,
	typename BfsEnactor,
	typename ProblemStorage,
	typename VertexId,
	typename Value,
	typename SizeT>
cudaError_t TestGpuBfs(
	BfsEnactor 								&enactor,
	ProblemStorage 							&csr_problem,
	VertexId 								src,
	VertexId 								*h_source_path,						// place to copy results out to
	VertexId 								*reference_source_dist,
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,							// reference host graph
	Stats									&stats,								// running statistics
	int 									max_grid_size)
{
	cudaError_t retval;

	do {

		// (Re)initialize distances
		if (retval = csr_problem.Reset()) break;

		// Perform BFS
		GpuTimer gpu_timer;
		gpu_timer.Start();

		if (retval = enactor.template EnactSearch<INSTRUMENT>(
			csr_problem,
			src,
			csr_graph.row_offsets[src],
			csr_graph.row_offsets[src + 1] - csr_graph.row_offsets[src],
			max_grid_size)) break;

		gpu_timer.Stop();
		float elapsed = gpu_timer.ElapsedMillis();

		// Copy out results
		if (retval = csr_problem.ExtractResults(h_source_path)) break;

		long long 	total_queued = 0;
		VertexId	search_depth = 0;
		double		expand_duty = 0.0;
		double		compact_duty = 0.0;

		enactor.GetStatistics(total_queued, search_depth, expand_duty, compact_duty);

		DisplayStats<ProblemStorage::ProblemType::MARK_PARENTS>(
			stats,
			src,
			h_source_path,
			reference_source_dist,
			csr_graph,
			elapsed,
			search_depth,
			total_queued,
			expand_duty,
			compact_duty);

	} while (0);
	
	return retval;
}


/**
 * A simple CPU-based reference BFS ranking implementation.  
 * 
 * Computes the distance of each node from the specified source node. 
 */
template<
	typename VertexId,
	typename Value,
	typename SizeT>
void SimpleReferenceBfs(
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,
	VertexId 								*source_path,
	VertexId 								src,
	Stats									&stats)								// running statistics
{
	// (Re)initialize distances
	for (VertexId i = 0; i < csr_graph.nodes; i++) {
		source_path[i] = -1;
	}
	source_path[src] = 0;
	VertexId search_depth = 0;

	// Initialize queue for managing previously-discovered nodes
	std::deque<VertexId> frontier;
	frontier.push_back(src);

	//
	// Perform BFS 
	//
	
	CpuTimer cpu_timer;
	cpu_timer.Start();
	while (!frontier.empty()) {
		
		// Dequeue node from frontier
		VertexId dequeued_node = frontier.front();
		frontier.pop_front();
		VertexId neighbor_dist = source_path[dequeued_node] + 1;

		// Locate adjacency list
		int edges_begin = csr_graph.row_offsets[dequeued_node];
		int edges_end = csr_graph.row_offsets[dequeued_node + 1];

		for (int edge = edges_begin; edge < edges_end; edge++) {

			// Lookup neighbor and enqueue if undiscovered 
			VertexId neighbor = csr_graph.column_indices[edge];
			if (source_path[neighbor] == -1) {
				source_path[neighbor] = neighbor_dist;
				if (search_depth < neighbor_dist) {
					search_depth = neighbor_dist;
				}
				frontier.push_back(neighbor);
			}
		}
	}
	cpu_timer.Stop();
	float elapsed = cpu_timer.ElapsedMillis();
	search_depth++;

//	Histogram(src, source_path, csr_graph, search_depth);

	DisplayStats<false, VertexId, Value, SizeT>(
		stats,
		src,
		source_path,
		NULL,						// No reference source path
		csr_graph,
		elapsed,
		search_depth,
		0,							// No redundant queuing
		0,
		0);							// No barrier duty
}


/**
 * Runs tests
 */
template <
	typename VertexId,
	typename Value,
	typename SizeT,
	bool INSTRUMENT,
	bool MARK_PARENTS>
void RunTests(
	const CsrGraph<VertexId, Value, SizeT> &csr_graph,
	VertexId src,
	bool randomized_src,
	int test_iterations,
	int max_grid_size,
	int num_gpus,
	double queue_sizing,
	bool stream_from_host)
{
	// Allocate host-side source_distance array (for both reference and gpu-computed results)
	VertexId* reference_source_dist 	= (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
	VertexId* h_source_path 			= (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);

	// Allocate a BFS enactor (with maximum frontier-queue size the size of the edge-list)
	bfs::microbench::EnactorGatherLookup micobench_enactor(g_verbose);

	// Allocate problem on GPU
	bfs::CsrProblem<VertexId, SizeT, MARK_PARENTS> csr_problem;
	if (csr_problem.FromHostProblem(
		stream_from_host,
		csr_graph.nodes,
		csr_graph.edges,
		csr_graph.column_indices,
		csr_graph.row_offsets,
		queue_sizing,
		g_uneven,
		num_gpus))
	{
		exit(1);
	}

	// Initialize statistics
	Stats stats[2];
	stats[0] = Stats("Simple CPU BFS");
	stats[1] = Stats("Microbench GPU BFS");
	
	printf("Running %s %s %s tests...\n\n",
		(INSTRUMENT) ? "instrumented" : "non-instrumented",
		(MARK_PARENTS) ? "parent-marking" : "distance-marking",
		(stream_from_host) ? "stream-from-host" : "copied-to-device");
	fflush(stdout);
	
	// Perform the specified number of test iterations
	int test_iteration = 0;
	while (test_iteration < test_iterations) {
	
		// If randomized-src was specified, re-roll the src
		if (randomized_src) src = builder::RandomNode(csr_graph.nodes);
		
		printf("---------------------------------------------------------------\n");

		// Compute reference CPU BFS solution for source-distance
		if (!g_quick) {
			SimpleReferenceBfs(csr_graph, reference_source_dist, src, stats[0]);
			printf("\n");
			fflush(stdout);
		}

		if (num_gpus == 1) {
			// Perform two-phase out-of-core BFS implementation (BFS level grid launch)
			if (TestGpuBfs<INSTRUMENT>(
				micobench_enactor,
				csr_problem,
				src,
				h_source_path,
				(g_quick) ? (VertexId*) NULL : reference_source_dist,
				csr_graph,
				stats[1],
				max_grid_size)) exit(1);
			printf("\n");
			fflush(stdout);
		}

		if (g_verbose2) {
			printf("Reference solution: ");
			DisplaySolution(reference_source_dist, csr_graph.nodes);
			printf("Computed solution (%s): ", (MARK_PARENTS) ? "parents" : "source dist");
			DisplaySolution(h_source_path, csr_graph.nodes);
			printf("\n");
		}
		
		if (randomized_src) {
			// Valid test_iterations is the maximum of any of the
			// test-statistic-structure's sample-counts
			test_iteration = 0;
			for (int i = 0; i < sizeof(stats) / sizeof(Stats); i++) {
				if (stats[i].rate.count > test_iteration) {
					test_iteration = stats[i].rate.count;
				}
			}
		} else {
			test_iteration++;
		}
	}
	
	
	//
	// Cleanup
	//
	
	if (reference_source_dist) free(reference_source_dist);
	if (h_source_path) free(h_source_path);

	cudaDeviceSynchronize();
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)  
{
	typedef int VertexId;							// Use as the node identifier type
	typedef int Value;								// Use as the value type
	typedef int SizeT;								// Use as the graph size type
	
	VertexId 	src 				= -1;			// Use whatever the specified graph-type's default is
	char* 		src_str				= NULL;
	bool 		randomized_src		= false;		// Whether or not to select a new random src for each test iteration
	bool 		instrumented		= false;		// Whether or not to collect instrumentation from kernels
	bool 		mark_parents		= false;		// Whether or not to mark src-distance vs. parent vertices
	bool		stream_from_host	= false;		// Whether or not to stream CSR representation from host mem
	int 		test_iterations 	= 1;
	int 		max_grid_size 		= 0;			// Maximum grid size (0: leave it up to the enactor)
	int 		num_gpus			= 1;			// Number of GPUs for multi-gpu enactor to use
	double 		queue_sizing		= 0.0;			// Scaling factor for work queues (0.0: leave it up to CsrProblemType)

	CommandLineArgs args(argc, argv);
	DeviceInit(args);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	srand(0);									// Presently deterministic
	//srand(time(NULL));	

	//
	// Check command line arguments
	// 
	
	if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 1;
	}
	instrumented = args.CheckCmdLineFlag("instrumented");
	args.GetCmdLineArgument("src", src_str);
	if (src_str != NULL) {
		if (strcmp(src_str, "randomize") == 0) {
			randomized_src = true;
		} else {
			src = atoi(src_str);
		}
	}
	g_undirected = args.CheckCmdLineFlag("undirected");
	g_quick = args.CheckCmdLineFlag("quick");
	mark_parents = args.CheckCmdLineFlag("mark-parents");
	stream_from_host = args.CheckCmdLineFlag("stream-from-host");
	g_uneven = args.CheckCmdLineFlag("uneven");
	args.GetCmdLineArgument("i", test_iterations);
	args.GetCmdLineArgument("max-ctas", max_grid_size);
	args.GetCmdLineArgument("num-gpus", num_gpus);
	args.GetCmdLineArgument("queue-sizing", queue_sizing);
	if (g_verbose2 = args.CheckCmdLineFlag("v2")) {
		g_verbose = true;
	} else {
		g_verbose = args.CheckCmdLineFlag("v");
	}
	int flags = args.ParsedArgc();
	int graph_args = argc - flags - 1;

	// Enable symmetric peer access between gpus
	for (int gpu = 0; gpu < num_gpus; gpu++) {
		for (int other_gpu = (gpu + 1) % num_gpus;
			other_gpu != gpu;
			other_gpu = (other_gpu + 1) % num_gpus)
		{
			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			printf("Enabling peer access to GPU %d from GPU %d\n", other_gpu, gpu);

			cudaError_t error = cudaDeviceEnablePeerAccess(other_gpu, 0);
			if ((error != cudaSuccess) && (error != cudaErrorPeerAccessAlreadyEnabled)) {
				util::B40CPerror(error, "MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__);
				exit(1);
			}
		}
	}


	//
	// Obtain CSR search graph
	//

	CsrGraph<VertexId, Value, SizeT> csr_graph(stream_from_host);
	
	if (graph_args < 1) {
		Usage();
		return 1;
	}
	std::string graph_type = argv[1];
	if (graph_type == "grid2d") {
		// Two-dimensional regular lattice grid (degree 4)
		if (graph_args < 2) { Usage(); return 1; }
		VertexId width = atoi(argv[2]);
		if (builder::BuildGrid2dGraph<false>(width, src, csr_graph) != 0) {
			return 1;
		}

	} else if (graph_type == "grid3d") {
		// Three-dimensional regular lattice grid (degree 6)
		if (graph_args < 2) { Usage(); return 1; }
		VertexId width = atoi(argv[2]);
		if (builder::BuildGrid3dGraph<false>(width, src, csr_graph) != 0) {
			return 1;
		}

	} else if (graph_type == "dimacs") {
		// DIMACS-formatted graph file
		if (graph_args < 1) { Usage(); return 1; }
		char *dimacs_filename = (graph_args == 2) ? argv[2] : NULL;
		bool splice = args.CheckCmdLineFlag("splice");
		if (builder::BuildDimacsGraph<false>(
			dimacs_filename,
			src,
			csr_graph,
			g_undirected,
			splice) != 0)
		{
			return 1;
		}
		
	} else if (graph_type == "metis") {
		// METIS-formatted graph file
		if (graph_args < 1) { Usage(); return 1; }
		char *metis_filename = (graph_args == 2) ? argv[2] : NULL;
		if (builder::BuildMetisGraph<false>(metis_filename, src, csr_graph) != 0) {
			return 1;
		}
		
	} else if (graph_type == "market") {
		// Matrix-market coordinate-formatted graph file
		if (graph_args < 1) { Usage(); return 1; }
		char *market_filename = (graph_args == 2) ? argv[2] : NULL;
		if (builder::BuildMarketGraph<false>(market_filename, src, csr_graph) != 0) {
			return 1;
		}

	} else if (graph_type == "random") {
		// Random graph of n nodes and m edges
		if (graph_args < 3) { Usage(); return 1; }
		SizeT nodes = atol(argv[2]);
		SizeT edges = atol(argv[3]);
		if (builder::BuildRandomGraph<false>(nodes, edges, src, csr_graph, g_undirected) != 0) {
			return 1;
		}

	} else if (graph_type == "rr") {
		// Random-regular-ish graph of n nodes, each with degree d (allows loops and cycles)
		if (graph_args < 3) { Usage(); return 1; }
		SizeT nodes = atol(argv[2]);
		int degree = atol(argv[3]);
		if (builder::BuildRandomRegularishGraph<false>(nodes, degree, src, csr_graph) != 0) {
			return 1;
		}

	} else {
		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;
	}
	
	// Optionally display graph
	if (g_verbose2) {
		printf("\n");
		csr_graph.DisplayGraph();
		printf("\n");
	}
	csr_graph.PrintHistogram();

	// Run tests

	if (instrumented) {
		// Run instrumented kernel for runtime statistics
		if (mark_parents) {
			RunTests<VertexId, Value, SizeT, true, true>(
				csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, queue_sizing, stream_from_host);
		} else {
			RunTests<VertexId, Value, SizeT, true, false>(
				csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, queue_sizing, stream_from_host);
		}
	} else {

		// Run regular kernel 
		if (mark_parents) {
			RunTests<VertexId, Value, SizeT, false, true>(
				csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, queue_sizing, stream_from_host);
		} else {
			RunTests<VertexId, Value, SizeT, false, false>(
				csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, queue_sizing, stream_from_host);
		}
	}
}
