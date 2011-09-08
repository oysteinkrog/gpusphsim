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
#include <b40c/copy/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

#pragma warning(disable : 4344)

using namespace b40c;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/


/**
 * Example showing syntax for invoking templated member functions from 
 * a templated function
 */
template <typename T, b40c::copy::ProbSizeGenre PROBLEM_SIZE_GENRE>
void TemplatedSubroutineCopy(
	b40c::copy::Enactor &copy_enactor,
	T *d_dest, 
	T *d_src,
	int num_elements)
{
	copy_enactor.template Copy<PROBLEM_SIZE_GENRE>(d_dest, d_src, num_elements * sizeof(T));
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_copy [--device=<device index>]\n");
    	return 0;
    }

	DeviceInit(args);

	typedef unsigned int T;
	const int NUM_ELEMENTS = 10;

	// Allocate and initialize host data
	T h_src[NUM_ELEMENTS];
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		h_src[i] = i;
	}
	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);
	cudaMemcpy(d_src, h_src, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	
	// Create a copy enactor
	b40c::copy::Enactor copy_enactor;
	
	//
	// Example 1: Enact simple copy using internal tuning heuristics
	//
	copy_enactor.Copy(d_dest, d_src, NUM_ELEMENTS * sizeof(T));
	
	printf("Simple copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");
	
	
	//
	// Example 2: Enact simple copy using "large problem" tuning configuration
	//
	copy_enactor.Copy<b40c::copy::LARGE_SIZE>(d_dest, d_src, NUM_ELEMENTS * sizeof(T));

	printf("Large-tuned copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 3: Enact simple copy using "small problem" tuning configuration
	//
	copy_enactor.Copy<b40c::copy::SMALL_SIZE>(d_dest, d_src, NUM_ELEMENTS * sizeof(T));
	
	printf("Small-tuned copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 4: Enact simple copy using a templated subroutine function
	//
	TemplatedSubroutineCopy<T, b40c::copy::UNKNOWN_SIZE>(copy_enactor, d_dest, d_src, NUM_ELEMENTS);
	
	printf("Templated subroutine copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 5: Enact simple copy using custom tuning configuration (base copy enactor)
	//
	typedef b40c::copy::Policy<
		T, 
		unsigned long long,
		b40c::copy::SM20, 
		8, 8, 7, 1, 0,
		b40c::util::io::ld::cg,
		b40c::util::io::st::cs,
		true, 
		false> CustomPolicy;
	
	copy_enactor.Copy<CustomPolicy>(d_dest, d_src, NUM_ELEMENTS);

	printf("Custom copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	return 0;
}

