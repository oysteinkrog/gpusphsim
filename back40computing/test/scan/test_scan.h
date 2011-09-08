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
 * Simple test utilities for scan
 ******************************************************************************/

#pragma once

#include <stdio.h> 

// Scan includes
#include <b40c/scan/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Test wrappers for binary, associative operations
 ******************************************************************************/

template <typename T>
struct Sum
{
	// Binary reduction
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}

	static const bool IS_COMMUTATIVE = true;
};

template <typename T>
struct Max
{
	// Binary reduction
	__host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}

	static const bool IS_COMMUTATIVE = true;
};


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Timed scan.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	bool EXCLUSIVE,
	b40c::scan::ProbSizeGenre PROB_SIZE_GENRE,
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
double TimedScan(
	T *h_data,
	T *h_reference,
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op,
	int max_ctas,
	bool verbose,
	int iterations)
{
	using namespace b40c;

	// Allocate device storage
	T *d_src, *d_dest;
	if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedScan cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedScan cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Create enactor
	scan::Enactor scan_enactor;

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedScan cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	printf("\n");
	scan_enactor.ENACTOR_DEBUG = true;
	scan_enactor.template Scan<EXCLUSIVE, ReductionOp::IS_COMMUTATIVE, PROB_SIZE_GENRE>(
		d_dest, d_src, num_elements, scan_op, identity_op, max_ctas);
	scan_enactor.ENACTOR_DEBUG = false;

	// Perform the timed number of iterations
	b40c::GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < iterations; i++) {

		// Start timing record
		timer.Start();

		// Call the scan API routine
		scan_enactor.template Scan<EXCLUSIVE, ReductionOp::IS_COMMUTATIVE, PROB_SIZE_GENRE>(
			d_dest, d_src, num_elements, scan_op, identity_op, max_ctas);

		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	printf("\nB40C %s scan: %d iterations, %lu elements, ",
		EXCLUSIVE ? "exclusive" : "inclusive", iterations, (unsigned long) num_elements);
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, throughput * sizeof(T) * 3);

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
	if (verbose) {
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


