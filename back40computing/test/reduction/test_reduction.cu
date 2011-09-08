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
 * Simple test driver program for reduction.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_reduction.h"

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
	printf("\ntest_reduction [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--sweep]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the reduction operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}


/**
 * Creates an example reduction problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	typename SizeT,
	typename ReductionOp>
void TestReduction(
	SizeT num_elements,
	ReductionOp reduction_op)
{
    // Allocate the reduction problem on the host and fill the keys with random bytes

	T *h_data 			= (T*) malloc(num_elements * sizeof(T));
	T *h_reference 		= (T*) malloc(sizeof(T));

	if ((h_data == NULL) || (h_reference == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (size_t i = 0; i < num_elements; ++i) {
		// util::RandomBits<T>(h_data[i], 0);
		h_data[i] = i;
		h_reference[0] = (i == 0) ?
			h_data[i] :
			reduction_op(h_reference[0], h_data[i]);
	}

	//
    // Run the timing test(s)
	//


	// Execute test(s), optionally sweeping problem size downward
	size_t orig_num_elements = num_elements;
	do {
		printf("\nLARGE config:\t");
		double large = TimedReduction<reduction::LARGE_SIZE>(
			h_data, h_reference, num_elements, reduction_op, g_max_ctas, g_verbose, g_iterations);

		printf("\nSMALL config:\t");
		double small = TimedReduction<reduction::SMALL_SIZE>(
			h_data, h_reference, num_elements, reduction_op, g_max_ctas, g_verbose, g_iterations);

		if (small > large) {
			printf("%lu-byte elements: Small faster at %lu elements\n", (unsigned long) sizeof(T), (unsigned long) num_elements);
		}

		num_elements -= 4096;

	} while (g_sweep && (num_elements < orig_num_elements ));

	// Free our allocated host memory
	if (h_data) free(h_data);
    if (h_reference) free(h_reference);
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
		Sum<T> reduction_op;
    	TestReduction<T>(num_elements * 4, reduction_op);
	}
	{
		printf("\n-- UNSIGNED SHORT ----------------------------------------------\n");
		typedef unsigned short T;
		Sum<T> reduction_op;
    	TestReduction<T>(num_elements * 2, reduction_op);
	}
	{
		printf("\n-- UNSIGNED INT -----------------------------------------------\n");
		typedef unsigned int T;
		Sum<T> reduction_op;
    	TestReduction<T>(num_elements, reduction_op);
	}
	{
		printf("\n-- UNSIGNED LONG LONG -----------------------------------------\n");
		typedef unsigned long long T;
		Sum<T> reduction_op;
    	TestReduction<T>(num_elements / 2, reduction_op);
	}

	return 0;
}



