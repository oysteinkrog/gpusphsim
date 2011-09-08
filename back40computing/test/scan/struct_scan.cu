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
#include <b40c/scan/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Struct for doing addition and max scan simultaneously
 */
struct Foo
{
	int a, b;

	__host__ __device__ __forceinline__ Foo() :
		a(0), b(0) {}

	__host__ __device__ __forceinline__ Foo(int a, int b) :
		a(a), b(b) {}

	__host__ __device__ __forceinline__ bool operator == (const Foo& other) const
	{
		return ((a == other.a) && (b == other.b));
	}

	__host__ __device__ __forceinline__ bool operator != (const Foo& other) const
	{
		return ((a != other.a) || (b != other.b));
	}

	void Print()
	{
		printf("[a: %d, b: %d]", a, b);
	}
};


/**
 * Foo binary scan operator
 */
struct MultiScan
{
	// Associative reduction operator
	__host__ __device__ __forceinline__ Foo operator()(const Foo &x, const Foo &y)
	{
		return Foo(
			x.a + y.a,
			(x.b > y.b) ? x.b : y.b);
	}

	// Identity operator
	__host__ __device__ __forceinline__ Foo operator()()
	{
		return Foo();
	}

	enum {
		NON_COMMUTATIVE = true,
	};
};


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_scan [--device=<device index>]\n");
    	return 0;
    }

    b40c::DeviceInit(args);
    int num_elements = 564;
    bool verbose = args.CheckCmdLineFlag("v");
    bool exclusive = args.CheckCmdLineFlag("exclusive");
    args.GetCmdLineArgument("n", num_elements);

	// Allocate and initialize host problem data and host reference solution
	Foo *h_src = new Foo[num_elements];
	Foo *h_reference = new Foo[num_elements];
	MultiScan max_op;

	for (size_t i = 0; i < num_elements; ++i) {
		h_src[i] = Foo(i, i);

		if (exclusive) {
			h_reference[i] = (i == 0) ?
				max_op() :									// identity
				max_op(h_reference[i - 1], h_src[i - 1]);
		} else {
			h_reference[i] = (i == 0) ?
				h_src[i] :
				max_op(h_reference[i - 1], h_src[i]);
		}
	}

	
	// Allocate and initialize device data
	Foo *d_src, *d_dest;
	cudaMalloc((void**) &d_src, sizeof(Foo) * num_elements);
	cudaMalloc((void**) &d_dest, sizeof(Foo) * num_elements);
	cudaMemcpy(d_src, h_src, sizeof(Foo) * num_elements, cudaMemcpyHostToDevice);


	// Create a scan enactor
	b40c::scan::Enactor scan_enactor;

	// Enact simple exclusive scan using internal tuning heuristics
	if (exclusive) {
		scan_enactor.Scan<true, MultiScan::NON_COMMUTATIVE>(
			d_dest, d_src, num_elements, max_op, max_op);
	} else {
		scan_enactor.Scan<false, MultiScan::NON_COMMUTATIVE>(
			d_dest, d_src, num_elements, max_op, max_op);
	}
	
	printf("Simple scan: "); b40c::CompareDeviceResults(h_reference, d_dest, num_elements, verbose, verbose); printf("\n");

	delete h_src;
	delete h_reference;

	return 0;
}

