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

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool 	g_verbose 						= false;
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
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>]\n");
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
 * Timed Thrust reduction.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	typename T,
	typename SizeT,
	typename ReductionOp>
double TimedThrustReduction(
	T *h_data,
	T *h_reference,
	SizeT num_elements,
	ReductionOp reduction_op)
{
	using namespace b40c;

	T h_dest[1] = {0};
	
	// Allocate device storage  
	T *d_src, *d_dest;
	if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedReduction cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T)),
		"TimedReduction cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedReduction cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);
	
	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	thrust::device_ptr<T> dev_ptr(d_src);		
	h_dest[0] = thrust::reduce(dev_ptr, dev_ptr + num_elements, (T) 0, reduction_op);
	
	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start timing record
		timer.Start();

		h_dest[0] = thrust::reduce(dev_ptr, dev_ptr + num_elements, (T) 0, reduction_op);
		
		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	printf("\nThrust Reduction: %d iterations, %lu elements, ", g_iterations, (unsigned long) num_elements);
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, throughput * sizeof(T));
	
    // Free allocated memory
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Display copied data
	if (g_verbose) {
		printf("Reduction: ");
		PrintValue(h_dest[0]);
		printf(", Reference: ");
		PrintValue(h_reference[0]);
		printf("\n\n");
	}

    // Verify solution
	CompareResults(h_dest, h_reference, 1, true);
	printf("\n");
	fflush(stdout);

	return throughput;
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

	double b40c = TimedReduction<reduction::UNKNOWN_SIZE>(
		h_data, h_reference, num_elements, reduction_op, g_max_ctas, g_verbose, g_iterations);

	double thrust = TimedThrustReduction(
		h_data, h_reference, num_elements, reduction_op);

	printf("B40C speedup: %.2f\n", b40c/thrust);

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
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	// Execute test(s)
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



