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
 * Simple test driver program for copy.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_copy.h"

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool 	g_verbose 						= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
int 	g_num_elements					= 1024;



/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_copy [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-bytes>]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the copy operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = %d\n", g_iterations);
	printf("\n");
	printf("\t--n\tThe number of bytes to comprise the sample problem\n");
	printf("\t\t\tDefault = %lu\n", (unsigned long) g_num_elements);
	printf("\n");
}


/**
 * Timed Runtime copy.  Uses the GPU to copy the specified vector of bytes for the given
 * number of iterations, displaying runtime information.
 */
template <typename T>
double TimedRuntimeCopy(
	T *d_src,
	T *d_dest,
	T *h_reference,
	size_t num_elements)
{
	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	cudaMemcpy(d_dest, d_src, sizeof(T) * num_elements, cudaMemcpyDeviceToDevice);
	
	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start timing record
		timer.Start();

		cudaMemcpy(d_dest, d_src, sizeof(T) * num_elements, cudaMemcpyDeviceToDevice);
		
		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	printf("\nRuntime Copy: %d iterations, %lu bytes, ", g_iterations, (unsigned long) num_elements);
    printf("%f GPU ms, %f x10^9 B/sec, ",
		avg_runtime, throughput * sizeof(T) * 2);
	
    // Copy out data
	T *h_dest = (T*) malloc(num_elements * sizeof(T));
    if (util::B40CPerror(cudaMemcpy(h_dest, d_dest, sizeof(T) * num_elements, cudaMemcpyDeviceToHost),
		"TimedScan cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Display copied data
	if (g_verbose) {
		printf("\n\nData:\n");
		for (int i = 0; i < num_elements; i++) {
			PrintValue<T>(h_dest[i]);
			printf(", ");
		}
		printf("\n\n");
	}

    // Verify solution
	CompareResults(h_dest, h_reference, num_elements, true);
	printf("\n");
	fflush(stdout);

	if (h_dest) free(h_dest);

	return throughput;
}


/**
 * Creates an example copy problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
void TestCopy(size_t num_elements)
{
	typedef unsigned char T;

    // Allocate the copy problem on the host and fill the keys with random bytes

	T *h_data 			= (T*) malloc(num_elements * sizeof(T));
	T *h_reference 		= (T*) malloc(num_elements * sizeof(T));

	if ((h_data == NULL) || (h_reference == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (size_t i = 0; i < num_elements; ++i) {
		// util::RandomBits<T>(h_data[i], 0);
		h_data[i] = i;
		h_reference[i] = h_data[i];
	}

	// Allocate device storage
	T *d_src, *d_dest;
	if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedCopy cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedCopy cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedCopy cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	//
    // Run the timing test(s)
	//

	double b40c = TimedCopy<T, copy::UNKNOWN_SIZE>(
		d_src, d_dest, h_reference, num_elements, g_max_ctas, g_verbose, g_iterations);

	double runtime = TimedRuntimeCopy<T>(
		d_src, d_dest, h_reference, num_elements);

	printf("B40C speedup: %.2f\n", b40c/runtime);

    // Free allocated memory
	if (h_data) free(h_data);
    if (h_reference) free(h_reference);
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//srand(time(NULL));	
	srand(0);				// presently deterministic

    //
	// Check command line arguments
    //

    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", g_num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	// Execute test(s)
	TestCopy(g_num_elements);

	return 0;
}



