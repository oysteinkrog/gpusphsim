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
 * Simple test driver program for scan.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_scan.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool 	g_verbose 						= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
bool 	g_inclusive						= false;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_scan [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--inclusive]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the scan operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}





/**
 * Timed scan.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	bool EXCLUSIVE,
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
double TimedThrustScan(
	T *h_data,
	T *h_reference,
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op)
{
	using namespace b40c;

	// Allocate device storage  
	T *d_src, *d_dest;
	if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedScan cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedScan cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedScan cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);
	
	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	thrust::device_ptr<T> dev_src(d_src);
	thrust::device_ptr<T> dev_dest(d_dest);
	if (EXCLUSIVE) {
		thrust::exclusive_scan(dev_src, dev_src + num_elements, dev_dest);
	} else {
		thrust::inclusive_scan(dev_src, dev_src + num_elements, dev_dest);
	}
	
	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start timing record
		timer.Start();

		if (EXCLUSIVE) {
			thrust::exclusive_scan(dev_src, dev_src + num_elements, dev_dest);
		} else {
			thrust::inclusive_scan(dev_src, dev_src + num_elements, dev_dest);
		}
		
		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	printf("\nThrust Scan: %d iterations, %lu elements, ", g_iterations, (unsigned long) num_elements);
    printf("%f GPU ms, %f x10^9 elts/sec",
		avg_runtime, throughput);

    // Copy out data
	T *h_dest = (T*) malloc(num_elements * sizeof(T));
    if (util::B40CPerror(cudaMemcpy(h_dest, d_dest, sizeof(T) * num_elements, cudaMemcpyDeviceToHost),
		"TimedScan cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

    // Free allocated memory
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);

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
 * Creates an example scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	bool EXCLUSIVE,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
void TestScan(
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op)
{
    // Allocate the scan problem on the host and fill the keys with random bytes

	T *h_data 			= (T*) malloc(num_elements * sizeof(T));
	T *h_reference 		= (T*) malloc(num_elements * sizeof(T));

	if ((h_data == NULL) || (h_reference == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (size_t i = 0; i < num_elements; ++i) {
//		util::RandomBits<T>(h_data[i], 0);
		h_data[i] = i;
		if (EXCLUSIVE)
		{
			h_reference[i] = (i == 0) ?
				identity_op() :
				scan_op(h_reference[i - 1], h_data[i - 1]);
		} else {
			h_reference[i] = (i == 0) ?
				h_data[i] :
				scan_op(h_reference[i - 1], h_data[i]);
		}
	}

	//
    // Run the timing test(s)
	//

	double b40c = TimedScan<EXCLUSIVE, scan::UNKNOWN_SIZE>(
		h_data, h_reference, num_elements, scan_op, identity_op, g_max_ctas, g_verbose, g_iterations);

	double thrust = TimedThrustScan<EXCLUSIVE>(
		h_data, h_reference, num_elements, scan_op, identity_op);

	printf("B40C speedup: %.2f\n", b40c/thrust);
	

	// Free our allocated host memory 
	if (h_data) free(h_data);
    if (h_reference) free(h_reference);
}


/**
 * Creates an example scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
void TestScanVariety(
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op)
{
	if (g_inclusive) {
		TestScan<T, false>(num_elements, scan_op, identity_op);
	} else {
		TestScan<T, true>(num_elements, scan_op, identity_op);
	}
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

	size_t num_elements = 1024;

    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    g_inclusive = args.CheckCmdLineFlag("inclusive");
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	// Execute test(s)
	{
		printf("\n-- UNSIGNED CHAR ----------------------------------------------\n");
		typedef unsigned char T;
		Sum<T> op;
		TestScanVariety<T>(num_elements * 4, op, op);
	}
	{
		printf("\n-- UNSIGNED SHORT ----------------------------------------------\n");
		typedef unsigned short T;
		Sum<T> op;
		TestScanVariety<T>(num_elements * 2, op, op);
	}
	{
		printf("\n-- UNSIGNED INT -----------------------------------------------\n");
		typedef unsigned int T;
		Sum<T> op;
		TestScanVariety<T>(num_elements, op, op);
	}
	{
		printf("\n-- UNSIGNED LONG LONG -----------------------------------------\n");
		typedef unsigned long long T;
		Sum<T> op;
		TestScanVariety<T>(num_elements / 2, op, op);
	}

	return 0;
}



