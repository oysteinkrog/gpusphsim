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
#include <b40c/util/ping_pong_storage.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/radix_sort/problem_type.cuh>
#include <b40c/radix_sort/policy.cuh>
#include <b40c/radix_sort/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Problem / Tuning Policy Types
 ******************************************************************************/

/**
 * Sample sorting problem type (32-bit keys and 32-bit values)
 */
typedef b40c::radix_sort::ProblemType<
		unsigned int,						// Key type
		unsigned int,						// Value type (alternatively, use b40c::util::NullType for keys-only sorting)
		int> 								// SizeT (what type to use for counting)
	ProblemType;


/**
 * Sample radix sort tuning policy (for 32-bit keys and 32-bit values on Fermi)
 *
 * Downsweep Constraints:
 * 		(i) 	A load can't contain more than 256 keys or we might overflow inside a lane of
 * 				8-bit composite counters, i.e., (threads * load-vec-size <= 256), equivalently:
 *
 * 					(DOWNSWEEP_LOG_THREADS + DOWNSWEEP_LOG_LOAD_VEC_SIZE <= 8)
 *
 * 		(ii) 	We must have between one and one warp of raking threads per lane of composite
 * 				counters, i.e., (1 <= raking-threads / (loads-per-cycle * bins / 4) <= 32),
 * 				equivalently:
 *
 * 					(0 <= DOWNSWEEP_LOG_RAKING_THREADS - DOWNSWEEP_LOG_LOADS_PER_CYCLE - RADIX_BITS + 2 <= B40C_LOG_WARP_THREADS(arch))
 *
 * 		(iii) 	We must have more (or equal) threads than bins in the threadblock,
 * 				i.e., (threads >= bins) equivalently:
 *
 * 					DOWNSWEEP_LOG_THREADS >= RADIX_BITS
 *
 */
typedef b40c::radix_sort::Policy<
		ProblemType,				// Problem type

		// Common
		200,						// SM ARCH
		4,							// RADIX_BITS

		// Launch tuning policy
		10,							// LOG_SCHEDULE_GRANULARITY			The "grain" by which to divide up the problem input.  E.g., 7 implies a near-even distribution of 128-key chunks to each CTA.  Related to, but different from the upsweep/downswep tile sizes, which may be different from each other.
		b40c::util::io::ld::NONE,	// CACHE_MODIFIER					Load cache-modifier.  Valid values: NONE, ca, cg, cs
		b40c::util::io::st::NONE,	// CACHE_MODIFIER					Store cache-modifier.  Valid values: NONE, wb, cg, cs
		false,						// EARLY_EXIT						Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
		false,						// UNIFORM_SMEM_ALLOCATION			Whether or not to pad the dynamic smem allocation to ensure that all three kernels (upsweep, spine, downsweep) have the same overall smem allocation
		true, 						// UNIFORM_GRID_SIZE				Whether or not to launch the spine kernel with one CTA (all that's needed), or pad it up to the same grid size as the upsweep/downsweep kernels
		true,						// OVERSUBSCRIBED_GRID_SIZE			Whether or not to oversubscribe the GPU with CTAs, up to a constant factor (usually 4x the resident occupancy)

		// Policy for upsweep kernel.
		// 		Reduces/counts all the different digit numerals for a given digit-place
		//
		8,							// UPSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		7,							// UPSWEEP_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		0,							// UPSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2
		2,							// UPSWEEP_LOG_LOADS_PER_TILE		The number of loads (log) per tile.  Valid range: 0-2

		// Spine-scan kernel policy
		//		Prefix sum of upsweep histograms counted by each CTA.  Relatively insignificant in the grand scheme, not really worth tuning for large problems)
		//
		1,							// SPINE_CTA_OCCUPANCY				The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		7,							// SPINE_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		2,							// SPINE_LOG_LOAD_VEC_SIZE			The vector-load size (log) for each load (log).  Valid range: 0-2
		0,							// SPINE_LOG_LOADS_PER_TILE			The number of loads (log) per tile.  Valid range: 0-2
		5,							// SPINE_LOG_RAKING_THREADS			The number of raking threads (log) for local prefix sum.  Valid range: 5-SPINE_LOG_THREADS

		// Policy for downsweep kernel
		//		Given prefix counts, scans/scatters keys into appropriate bins
		// 		Note: a "cycle" is a tile sub-segment up to 256 keys
		//
		true,						// DOWNSWEEP_TWO_PHASE_SCATTER		Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
		8,							// DOWNSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		6,							// DOWNSWEEP_LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10, subject to constraints described above
		2,							// DOWNSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2, subject to constraints described above
		1,							// DOWNSWEEP_LOG_LOADS_PER_CYCLE	The number of loads (log) per cycle.  Valid range: 0-2, subject to constraints described above
		1, 							// DOWNSWEEP_LOG_CYCLES_PER_TILE	The number of cycles (log) per tile.  Valid range: 0-2
		6>							// DOWNSWEEP_LOG_RAKING_THREADS		The number of raking threads (log) for local prefix sum.  Valid range: 5-DOWNSWEEP_LOG_THREADS
	Policy;



/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
    typedef typename ProblemType::OriginalKeyType 	KeyType;
    typedef typename Policy::ValueType 				ValueType;
    typedef typename Policy::SizeT 					SizeT;

    // Initialize command line
    b40c::CommandLineArgs args(argc, argv);
    b40c::DeviceInit(args);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nlars_demo [--device=<device index>] [--v] [--n=<elements>] [--max-ctas=<max-thread-blocks>]\n");
    	return 0;
    }

    // Parse commandline args
    SizeT 			num_elements = 1024 * 1024 * 8;			// 8 million pairs
    unsigned int 	max_ctas = 0;							// default: let the enactor decide how many CTAs to launch based upon device properties

    bool verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", max_ctas);

	// Allocate and initialize host problem data and host reference solution
	KeyType *h_keys 				= new KeyType[num_elements];
	ValueType *h_values 			= new ValueType[num_elements];
	KeyType *h_reference_keys 		= new KeyType[num_elements];
	ValueType *h_reference_values 	= new ValueType[num_elements];

	// Only use RADIX_BITS effective bits (remaining high order bits
	// are left zero): we only want to perform one sorting pass
	printf("Original: ");
	for (size_t i = 0; i < num_elements; ++i) {
		b40c::util::RandomBits(h_keys[i], 0, Policy::RADIX_BITS);
		h_reference_keys[i] = h_keys[i];

		printf("%d, ", h_keys[i]);
	}
	printf("\n");

    // Compute reference solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);

	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);

	// Create a scan enactor
	b40c::radix_sort::Enactor enactor;

	// Create ping-pong storage wrapper.
	b40c::util::PingPongStorage<KeyType, ValueType> sort_storage(d_keys, d_values);

	//
	// Perform one sorting pass (starting at bit zero and covering RADIX_BITS bits)
	//

	cudaMemcpy(
		sort_storage.d_keys[sort_storage.selector],
		h_keys,
		sizeof(KeyType) * num_elements,
		cudaMemcpyHostToDevice);
	cudaMemcpy(
		sort_storage.d_values[sort_storage.selector],
		h_values,
		sizeof(ValueType) * num_elements,
		cudaMemcpyHostToDevice);

	enactor.Sort<
		0,
		Policy::RADIX_BITS,
		Policy>(sort_storage, num_elements, max_ctas);

	if (b40c::util::Equals<ValueType, b40c::util::NullType>::VALUE) {
		printf("Restricted-range keys-only sort: ");
	} else {
		printf("Restricted-range key-value sort: ");
	}
	b40c::CompareDeviceResults(
		h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

	// Cleanup any "pong" storage allocated by the enactor
	if (sort_storage.d_keys[1]) cudaFree(sort_storage.d_keys[1]);
	if (sort_storage.d_values[1]) cudaFree(sort_storage.d_values[1]);

	// Cleanup other
	delete h_keys;
	delete h_reference_keys;
	delete h_values;
	delete h_reference_values;

	return 0;
}

