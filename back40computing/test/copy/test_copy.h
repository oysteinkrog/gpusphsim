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
 * Simple test utilities for copy
 ******************************************************************************/

#pragma once

#include <stdio.h> 

// Copy includes
#include <b40c/copy/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Timed copy.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <typename T, b40c::copy::ProbSizeGenre PROB_SIZE_GENRE>
double TimedCopy(
	T *d_src,
	T *d_dest,
	T *h_reference,
	size_t num_elements,
	int max_ctas,
	bool verbose,
	int iterations,
	bool same_device = true)
{
	using namespace b40c;

	// Create enactor
	copy::Enactor copy_enactor;

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	copy_enactor.ENACTOR_DEBUG = true;
	copy_enactor.template Copy<PROB_SIZE_GENRE>(
		d_dest, d_src, num_elements * sizeof(T), max_ctas);
	copy_enactor.ENACTOR_DEBUG = false;

	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < iterations; i++) {

		// Start timing record
		timer.Start();

		// Call the copy API routine
		copy_enactor.template Copy<PROB_SIZE_GENRE>(
			d_dest, d_src, num_elements * sizeof(T), max_ctas);

		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	int bytes_per_element = (same_device) ? sizeof(T) * 2 : sizeof(T);
	printf("\nB40C copy: %d iterations, %lu bytes, ", iterations, (unsigned long) num_elements);
    printf("%f GPU ms, %f x10^9 B/sec, ",
		avg_runtime, throughput * bytes_per_element);

    // Copy out data
	T *h_dest = (T*) malloc(num_elements * sizeof(T));
    if (util::B40CPerror(cudaMemcpy(h_dest, d_dest, sizeof(T) * num_elements, cudaMemcpyDeviceToHost),
		"TimedScan cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

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



