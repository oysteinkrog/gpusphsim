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
 * Simple test driver program for radix sort.
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>

// Sorting includes
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/ping_pong_storage.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Constants
 ******************************************************************************/

const int LOWER_BITS = 17;


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_sort [--device=<device index>] [--v] [--n=<elements>] [--keys-only]\n");
    	return 0;
    }

    b40c::DeviceInit(args);
    unsigned int num_elements = 77;
    bool verbose = args.CheckCmdLineFlag("v");
    bool keys_only = args.CheckCmdLineFlag("keys-only");
    args.GetCmdLineArgument("n", num_elements);

	// Allocate and initialize host problem data and host reference solution
	int *h_keys = new int[num_elements];
	int *h_values = new int[num_elements];
	int *h_reference_keys = new int[num_elements];
	int *h_reference_values = new int[num_elements];

	for (size_t i = 0; i < num_elements; ++i) {
		b40c::util::RandomBits(h_keys[i], 0, LOWER_BITS);
		h_values[i] = i;
		h_reference_keys[i] = h_keys[i];
	}

    // Compute reference solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);

	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	int *d_keys;
	int *d_values;
	cudaMalloc((void**) &d_keys, sizeof(int) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(int) * num_elements);

	// Create a scan enactor
	b40c::radix_sort::Enactor enactor;

	if (keys_only) {

		// Keys-only sorting

		// Create ping-pong storage wrapper.
		b40c::util::PingPongStorage<int> sort_storage(d_keys);

		//
		// Example 1: simple sort.  Uses heuristics to select
		// appropriate problem-size-tuning granularity. (Using this
		// method causes the compiler to generate several tuning variants,
		// which can increase compilation times)
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort(sort_storage, num_elements);

		printf("Simple keys-only sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		//
		// Example 2: Small-problem-tuned sort.  Tuned for < 1M elements
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort<b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem keys-only sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		//
		// Example 3: small-problem-tuned sort over specific bit-range
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort<0, LOWER_BITS, b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem restricted-range keys-only sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		// Cleanup any "pong" storage allocated by the enactor
		if (sort_storage.d_keys[1]) cudaFree(sort_storage.d_keys[1]);

	} else {

		//
		// Key-value sorting
		//

		// Create ping-pong storage wrapper.
		b40c::util::PingPongStorage<int, int> sort_storage(d_keys, d_values);

		//
		// Example 1: simple sort.  Uses heuristics to select
		// appropriate problem-size-tuning granularity. (Using this
		// method causes the compiler to generate several tuning variants,
		// which can increase compilation times)
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(sort_storage.d_values[sort_storage.selector], h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort(sort_storage, num_elements);

		printf("Simple key-value sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		//
		// Example 2: Small-problem-tuned sort.  Tuned for < 1M elements
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(sort_storage.d_values[sort_storage.selector], h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort<b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem key-value sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");


		//
		// Example 3: small-problem-tuned sort over specific bit-range
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(sort_storage.d_values[sort_storage.selector], h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

		enactor.Sort<0, LOWER_BITS, b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem restricted-range key-value sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");


		// Cleanup any "pong" storage allocated by the enactor
		if (sort_storage.d_keys[1]) cudaFree(sort_storage.d_keys[1]);
		if (sort_storage.d_values[1]) cudaFree(sort_storage.d_values[1]);

	}
	
	delete h_keys;
	delete h_reference_keys;
	delete h_values;
	delete h_reference_values;

	return 0;
}

