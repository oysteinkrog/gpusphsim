/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for consecutive reduction.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_consecutive_reduction.h"

#include <b40c/util/ping_pong_storage.cuh>

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool 	g_verbose 						= false;
bool 	g_sweep							= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_consecutive_reduction [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--sweep]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the consecutive reduction operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}



/**
 * Creates an example consecutive reduction problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	typename SizeT,
	typename ReductionOp>
void TestConsecutiveReduction(
	SizeT num_elements,
	ReductionOp scan_op)
{
    // Allocate the consecutive reduction problem on the host
	typedef util::PingPongStorage<T, T> PingPongStorage;
	PingPongStorage h_problem_storage;

	h_problem_storage.d_keys[0] = (T*) malloc(num_elements * sizeof(T));
	h_problem_storage.d_keys[1] = (T*) malloc(num_elements * sizeof(T));
	h_problem_storage.d_values[0] = (T*) malloc(num_elements * sizeof(T));
	h_problem_storage.d_values[1] = (T*) malloc(num_elements * sizeof(T));

	if (!h_problem_storage.d_keys[0] || !h_problem_storage.d_keys[1] || !h_problem_storage.d_values[0] || !h_problem_storage.d_values[1]){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	// Initialize problem
	if (g_verbose) printf("Input problem: \n");
	for (int i = 0; i < num_elements; i++) {
		h_problem_storage.d_keys[0][i] = (i / 7) & 1;							// Toggle every 7 elements
//		util::RandomBits<T>(h_problem_storage.d_keys[0][i], 1, 1);				// Entropy-reduced random 0|1 values: roughly 26 / 64 elements toggled

		h_problem_storage.d_values[0][i] = 1;

		if (g_verbose) {
			printf("(%lld, %lld), ",
				(long long) h_problem_storage.d_keys[0][i],
				(long long) h_problem_storage.d_values[0][i]);
		}
	}
	if (g_verbose) printf("\n");

	// Compute reference solution
	SizeT num_compacted = 0;
	h_problem_storage.d_keys[1][0] = h_problem_storage.d_keys[0][0];

	for (SizeT i = 0; i < num_elements; ++i) {

		if (h_problem_storage.d_keys[1][num_compacted] != h_problem_storage.d_keys[0][i]) {

			num_compacted++;
			h_problem_storage.d_keys[1][num_compacted] = h_problem_storage.d_keys[0][i];
			h_problem_storage.d_values[1][num_compacted] = h_problem_storage.d_values[0][i];

		} else {

			if (i == 0) {
				h_problem_storage.d_values[1][num_compacted] =
					h_problem_storage.d_values[0][i];
			} else {
				h_problem_storage.d_values[1][num_compacted] = scan_op(
					h_problem_storage.d_values[1][num_compacted],
					h_problem_storage.d_values[0][i]);
			}
		}
	}
	num_compacted++;

	Equality<typename PingPongStorage::KeyType> equality_op;

	// Execute test(s), optionally sweeping problem size downward
	SizeT orig_num_elements = num_elements;
	do {

		printf("\nLARGE config:\t");
		double large = TimedConsecutiveReduction<consecutive_reduction::LARGE_SIZE>(
			h_problem_storage,
			num_elements,
			num_compacted,
			scan_op,
			equality_op,
			g_max_ctas,
			g_verbose,
			g_iterations);

		printf("\nSMALL config:\t");
		double small = TimedConsecutiveReduction<consecutive_reduction::SMALL_SIZE>(
			h_problem_storage,
			num_elements,
			num_compacted,
			scan_op,
			equality_op,
			g_max_ctas,
			g_verbose,
			g_iterations);

		if (small > large) {
			printf("%lu-byte elements: Small faster at %lu elements\n", (unsigned long) sizeof(T), (unsigned long) num_elements);
		}

		num_elements -= 4096;

	} while (g_sweep && (num_elements < orig_num_elements ));

	// Free our allocated host memory
	if (h_problem_storage.d_keys[0]) free(h_problem_storage.d_keys[0]);
	if (h_problem_storage.d_keys[1]) free(h_problem_storage.d_keys[1]);
	if (h_problem_storage.d_values[0]) free(h_problem_storage.d_values[0]);
	if (h_problem_storage.d_values[1]) free(h_problem_storage.d_values[1]);
}




/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	// Initialize commandline args and device
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Seed random number generator
	srand(0);				// presently deterministic
	//srand(time(NULL));

	// Use 32-bit integer for array indexing
	typedef int SizeT;
	SizeT num_elements = 1024;

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    g_sweep = args.CheckCmdLineFlag("sweep");
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	{
		printf("\n-- UNSIGNED CHAR ----------------------------------------------\n");
		typedef unsigned char T;
		Sum<T> op;
		TestConsecutiveReduction<T>(num_elements * 4, op);
	}
	{
		printf("\n-- UNSIGNED SHORT ----------------------------------------------\n");
		typedef unsigned short T;
		Sum<T> op;
		TestConsecutiveReduction<T>(num_elements * 2, op);
	}
	{
		printf("\n-- UNSIGNED INT -----------------------------------------------\n");
		typedef unsigned int T;
		Sum<T> op;
		TestConsecutiveReduction<T>(num_elements, op);
	}
	{
		printf("\n-- UNSIGNED LONG LONG -----------------------------------------\n");
		typedef unsigned long long T;
		Sum<T> op;
		TestConsecutiveReduction<T>(num_elements / 2, op);
	}

	return 0;
}



