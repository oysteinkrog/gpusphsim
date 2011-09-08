/******************************************************************************
 * 
 * Reductionright 2010-2011 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a reduction of the License at
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
#include <b40c/reduction/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Utility Routines
 ******************************************************************************/


/**
 * Max binary associative operator
 */
template <typename T>
struct Max
{
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}
};


/**
 * Example showing syntax for invoking templated member functions from 
 * a templated function
 */
template <
	typename T,
	typename SizeT,
	typename ReductionOp>
void TemplatedSubroutineReduction(
	b40c::reduction::Enactor &reduction_enactor,
	T *d_dest, 
	T *d_src,
	SizeT num_elements,
	ReductionOp reduction_op)
{
	reduction_enactor.template Reduce<T, SizeT, ReductionOp>(
		d_dest, d_src, num_elements, reduction_op);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_reduction [--device=<device index>]\n");
    	return 0;
    }

    b40c::DeviceInit(args);

	typedef unsigned int T;
	const int NUM_ELEMENTS = 34567;
	Max<T> max_op;

	// Allocate and initialize host problem data and host reference solution
	T *h_data = new T[NUM_ELEMENTS];
	T *h_reference = new T[1];
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		h_data[i] = i;
		h_reference[0] = (i == 0) ?
			h_data[i] :
			max_op(h_reference[0], h_data[i]);
	}
	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);
	cudaMemcpy(d_src, h_data, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	
	// Create a reduction enactor
	b40c::reduction::Enactor reduction_enactor;
	
	//
	// Example 1: Enact simple reduction using internal tuning heuristics
	//
	reduction_enactor.Reduce(d_dest, d_src, NUM_ELEMENTS, max_op);
	
	printf("Simple reduction: "); b40c::CompareDeviceResults(h_reference, d_dest, 1); printf("\n");
	

	//
	// Example 2: Enact simple reduction using "large problem" tuning configuration
	//
	reduction_enactor.Reduce<b40c::reduction::LARGE_SIZE>(
		d_dest, d_src, NUM_ELEMENTS, max_op);

	printf("Large-tuned reduction: "); b40c::CompareDeviceResults(h_reference, d_dest, 1); printf("\n");


	//
	// Example 3: Enact simple reduction using "small problem" tuning configuration
	//
	reduction_enactor.Reduce<b40c::reduction::SMALL_SIZE>(
		d_dest, d_src, NUM_ELEMENTS, max_op);
	
	printf("Small-tuned reduction: "); b40c::CompareDeviceResults(h_reference, d_dest, 1); printf("\n");


	//
	// Example 4: Enact simple reduction using a templated subroutine function
	//
	TemplatedSubroutineReduction(reduction_enactor, d_dest, d_src, NUM_ELEMENTS, max_op);
	
	printf("Templated subroutine reduction: "); b40c::CompareDeviceResults(h_reference, d_dest, 1); printf("\n");


	//
	// Example 5: Enact simple reduction using custom tuning configuration (base reduction enactor)
	//

	typedef Max<T> ReductionOp;
	typedef b40c::reduction::ProblemType<T, int, ReductionOp> ProblemType;
	typedef b40c::reduction::Policy<
		ProblemType,
		b40c::reduction::SM20,
		b40c::util::io::ld::cg,
		b40c::util::io::st::cg,
		true,
		false,
		true, 
		false, 
		8, 7, 1, 2, 9,
		8, 1, 1> CustomPolicy;
	
	reduction_enactor.Reduce<CustomPolicy>(d_dest, d_src, NUM_ELEMENTS, max_op);

	printf("Custom reduction: "); b40c::CompareDeviceResults(h_reference, d_dest, 1); printf("\n");

	delete h_data;
	delete h_reference;

	return 0;
}

