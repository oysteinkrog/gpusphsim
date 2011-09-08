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

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

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
	printf("\ncompare_segmented_scan [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--inclusive]\n");
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


template <typename OutputType, typename HeadFlagType, typename AssociativeOperator>
struct segmented_scan_functor
{
    AssociativeOperator binary_op;

    typedef typename thrust::tuple<OutputType, HeadFlagType> result_type;

    __host__ __device__
    segmented_scan_functor(AssociativeOperator _binary_op) : binary_op(_binary_op) {}

    __host__ __device__
    result_type operator()(result_type a, result_type b)
    {
        return result_type(thrust::get<1>(b) ? thrust::get<0>(b) : binary_op(thrust::get<0>(a), thrust::get<0>(b)),
                           thrust::get<1>(a) | thrust::get<1>(b));
    }
};


/**
 * Timed consecutive reduction.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	typename PingPongStorage,
	typename SizeT>
double TimedThrustConsecutiveReduction(
	PingPongStorage &h_problem_storage,			// host problem storage (selector points to input, but output contains reference result)
	SizeT num_elements,
	SizeT num_compacted,						// number of elements in reference result
	int max_ctas,
	bool verbose,
	int iterations)
{
	using namespace b40c;

	typedef typename PingPongStorage::KeyType 		KeyType;
	typedef typename PingPongStorage::ValueType 	ValueType;

	// Allocate device storage
	PingPongStorage 	d_problem_storage;

	if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_keys[0], sizeof(KeyType) * num_elements),
		"TimedConsecutiveReduction cudaMalloc d_keys failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_keys[1], sizeof(KeyType) * num_elements),
		"TimedConsecutiveReduction cudaMalloc d_keys failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_values[0], sizeof(ValueType) * num_elements),
		"TimedConsecutiveReduction cudaMalloc d_values failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_values[1], sizeof(ValueType) * num_elements),
		"TimedConsecutiveReduction cudaMalloc d_values failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(
			d_problem_storage.d_keys[0],
			h_problem_storage.d_keys[0],
			sizeof(KeyType) * num_elements,
			cudaMemcpyHostToDevice),
		"TimedConsecutiveReduction cudaMemcpy d_keys failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMemcpy(
			d_problem_storage.d_values[0],
			h_problem_storage.d_values[0],
			sizeof(ValueType) * num_elements,
			cudaMemcpyHostToDevice),
		"TimedConsecutiveReduction cudaMemcpy d_values failed: ", __FILE__, __LINE__)) exit(1);

	thrust::device_ptr<KeyType> d_in_keys(d_problem_storage.d_keys[0]);
	thrust::device_ptr<KeyType> d_out_keys(d_problem_storage.d_keys[1]);
	thrust::device_ptr<ValueType> d_in_values(d_problem_storage.d_values[0]);
	thrust::device_ptr<ValueType> d_out_values(d_problem_storage.d_values[1]);

	thrust::pair<thrust::device_ptr<KeyType>, thrust::device_ptr<ValueType> > new_end;

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	new_end = thrust::reduce_by_key(
		d_in_keys,
		d_in_keys + num_elements,
		d_in_values,
		d_out_keys,
		d_out_values);

	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < iterations; i++) {

		// Start timing record
		timer.Start();

		// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
		new_end = thrust::reduce_by_key(
			d_in_keys,
			d_in_keys + num_elements,
			d_in_values,
			d_out_keys,
			d_out_values);

		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	long long bytes = ((num_elements * 2) + num_compacted) * (sizeof(KeyType) + sizeof(ValueType));
	double bandwidth = bytes / avg_runtime / 1000.0 / 1000.0;
	printf("\nThrust consecutive reduction: %d iterations, %lu elements -> %lu compacted, ",
		iterations, (unsigned long) num_elements, (unsigned long) num_compacted);
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, bandwidth);

	// Check and display results
	printf("\nCompacted Keys: ");
	CompareDeviceResults(h_problem_storage.d_keys[1], d_problem_storage.d_keys[1], num_compacted, verbose, verbose);
	printf("\nCompacted and reduced Values: ");
	CompareDeviceResults(h_problem_storage.d_values[1], d_problem_storage.d_values[1], num_compacted, verbose, verbose);

	printf("\nCompacted Size: %s",
			(num_compacted == (int) (new_end.first - d_out_keys)) ? "CORRECT" : "INCORRECT");

	printf("\n");
	fflush(stdout);

	// Free allocated memory
    if (d_problem_storage.d_keys[0]) cudaFree(d_problem_storage.d_keys[0]);
    if (d_problem_storage.d_keys[1]) cudaFree(d_problem_storage.d_keys[1]);
    if (d_problem_storage.d_values[0]) cudaFree(d_problem_storage.d_values[0]);
    if (d_problem_storage.d_values[1]) cudaFree(d_problem_storage.d_values[1]);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	return throughput;
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
    // Allocate the consecutive reduction problem on the host and fill the keys with random bytes
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
		h_problem_storage.d_keys[0][i] = (i / 7) & 1;		// Toggle every 7 elements
//		util::RandomBits<T>(h_data[i], 1, 1);				// Entropy-reduced random 0|1 values: roughly 26 / 64 elements toggled

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

	//
    // Run the timing test(s)
	//

	double b40c = TimedConsecutiveReduction<consecutive_reduction::UNKNOWN_SIZE>(
		h_problem_storage,
		num_elements,
		num_compacted,
		scan_op,
		equality_op,
		g_max_ctas,
		g_verbose,
		g_iterations);

	double thrust = TimedThrustConsecutiveReduction(
		h_problem_storage,
		num_elements,
		num_compacted,
		g_max_ctas,
		g_verbose,
		g_iterations);

	printf("B40C speedup: %.2f\n", b40c/thrust);

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
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	// Execute test(s)
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



