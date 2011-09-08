/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a scan of the License at
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

#include <b40c/segmented_scan/problem_type.cuh>
#include <b40c/segmented_scan/policy.cuh>
#include <b40c/segmented_scan/autotuned_policy.cuh>
#include <b40c/segmented_scan/enactor.cuh>

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Max binary scan operator
 */
template <typename T>
struct Max
{
	// Associative reduction operator
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	// Identity operator
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}
};



/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_copy [--device=<device index>]\n");
    	return 0;
    }

    b40c::DeviceInit(args);

    // Define our problem type
	typedef unsigned int T;
	typedef unsigned char Flag;

	const int NUM_ELEMENTS = 512;
	const bool EXCLUSIVE_SCAN = false;

	// Allocate and initialize host problem data and host reference solution
	T 		h_src[NUM_ELEMENTS];
	Flag 	h_flags[NUM_ELEMENTS];
	T 		h_reference[NUM_ELEMENTS];
	Max<T> 	max_op;

	for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
		h_src[i] = 1;
		h_flags[i] = (i % 11) == 0;
	}

	for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if (EXCLUSIVE_SCAN)
		{
			h_reference[i] = ((i == 0) || (h_flags[i])) ?
				max_op() :
				max_op(h_reference[i - 1], h_src[i - 1]);
		} else {
			h_reference[i] = ((i == 0) || (h_flags[i])) ?
				h_src[i] :
				max_op(h_reference[i - 1], h_src[i]);
		}
	}

	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	Flag *d_flags;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_flags, sizeof(Flag) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);

	cudaMemcpy(d_src, h_src, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_flags, h_flags, sizeof(Flag) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

	// Create enactor
	b40c::segmented_scan::Enactor segmented_scan_enactor;

	//
	// Example 1: Enact simple exclusive scan using internal tuning heuristics
	//
	segmented_scan_enactor.Scan<EXCLUSIVE_SCAN>(
		d_dest, d_src, d_flags, NUM_ELEMENTS, max_op, max_op);

	printf("Simple scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 2: Enact simple exclusive scan using "large problem" tuning configuration
	//
	segmented_scan_enactor.Scan<b40c::segmented_scan::LARGE_SIZE, EXCLUSIVE_SCAN>(
		d_dest, d_src, d_flags, NUM_ELEMENTS, max_op, max_op);

	printf("Large-tuned scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");

	//
	// Example 3: Custom segmented scan
	//

	typedef Max<T> ReductionOp;
	typedef Max<T> IdentityOp;
	typedef b40c::segmented_scan::ProblemType<T, Flag, int, ReductionOp, IdentityOp, EXCLUSIVE_SCAN> ProblemType;
	typedef b40c::segmented_scan::Policy<
		ProblemType,
		b40c::segmented_scan::SM20,
		b40c::util::io::ld::cg,
		b40c::util::io::st::cg,
		false, false, false, 7,
		8, 5, 2, 1, 5,
		5, 1, 1, 5,
		8, 5, 1, 2, 5> CustomConfig;

	segmented_scan_enactor.Scan<CustomConfig>(
		d_dest, d_src, d_flags, NUM_ELEMENTS, max_op, max_op);

	b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS);
	printf("Custom scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");

	printf("\n");



	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	return 0;
}

