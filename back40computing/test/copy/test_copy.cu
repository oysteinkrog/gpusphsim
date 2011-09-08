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

// Test utils
#include "b40c_test_util.h"
#include "test_copy.h"

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool 	g_verbose 						= false;
bool 	g_sweep							= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
size_t 	g_num_elements 					= 1024;
int 	g_src_gpu						= -1;
int 	g_dest_gpu						= -1;
bool 	g_from_host						= false;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_copy [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-bytes>] [--sweep] "
			"[ [--src=<src-gpu> --dest=<dest-gpu>] | --from-host ]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the copy operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = %d\n", g_iterations);
	printf("\n");
	printf("\t--n\tThe number of bytes to comprise the sample problem\n");
	printf("\t\t\tDefault = %lu\n", (unsigned long) g_num_elements);
	printf("\n");
}


/**
 * Creates an example copy problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
void TestCopy(size_t num_elements)
{
	typedef unsigned char T;

	// Allocate the copy problem on the host and fill the keys with random bytes

	T *h_data 			= (T*) malloc(num_elements * sizeof(T));
	T *h_reference 		= (T*) malloc(num_elements * sizeof(T));

	if ((h_data == NULL) || (h_reference == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (size_t i = 0; i < num_elements; ++i) {
		// util::RandomBits<T>(h_data[i], 0);
		h_data[i] = i;
		h_reference[i] = h_data[i];
	}

	// Allocate device storage (and leave g_dest_gpu as current gpu)
	T *h_src = NULL;
	T *d_src = NULL;
	T *d_dest = NULL;

	bool same_device = (!g_from_host) && (g_src_gpu == g_dest_gpu);

	if (g_from_host) {
		int flags = cudaHostAllocMapped;
		if (util::B40CPerror(cudaHostAlloc((void**) &h_src, sizeof(T) * num_elements, flags),
			"TimedCopy cudaHostAlloc d_src failed", __FILE__, __LINE__)) exit(1);

		// Map into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_src, (void *) h_src, 0),
			"TimedCopy cudaHostGetDevicePointer h_src failed", __FILE__, __LINE__)) exit(1);

	} else {
		if (util::B40CPerror(cudaSetDevice(g_src_gpu),
			"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
			"TimedCopy cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	}

	if (util::B40CPerror(cudaSetDevice(g_dest_gpu),
		"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedCopy cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedCopy cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	//
    // Run the timing test(s)
	//

	// Execute test(s), optionally sweeping problem size downward
	size_t orig_num_elements = num_elements;
	do {
		printf("\nLARGE config:\n");
		double large = TimedCopy<T, copy::LARGE_SIZE>(
			d_src, d_dest, h_reference, num_elements, g_max_ctas, g_verbose, g_iterations, same_device);

		printf("\nSMALL config:\n");
		double small = TimedCopy<T, copy::SMALL_SIZE>(
			d_src, d_dest, h_reference, num_elements, g_max_ctas, g_verbose, g_iterations, same_device);

		if (small > large) {
			printf("%lu-byte bytes: Small faster at %lu bytes\n", (unsigned long) sizeof(T), (unsigned long) num_elements);
		}

		num_elements -= 4096;

	} while (g_sweep && (num_elements < orig_num_elements ));

    // Free allocated memory
	if (h_data) free(h_data);
    if (h_reference) free(h_reference);
    if (h_src) {
		cudaFreeHost(h_src);
	} else {
		cudaFree(d_src);
	}
    if (d_dest) cudaFree(d_dest);
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

    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

	cudaSetDeviceFlags(cudaDeviceMapHost);

    g_sweep = args.CheckCmdLineFlag("sweep");
    g_from_host = args.CheckCmdLineFlag("from-host");
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", g_num_elements);
    args.GetCmdLineArgument("src", g_src_gpu);
    args.GetCmdLineArgument("dest", g_dest_gpu);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	if ((g_src_gpu > -1) && (g_dest_gpu > -1)) {

		printf("Inter-GPU copy.\n");

		// Set device
		if (util::B40CPerror(cudaSetDevice(g_src_gpu),
			"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
		printf("Enabling peer access to GPU %d from GPU %d\n", g_src_gpu, g_dest_gpu);
		if (util::B40CPerror(cudaDeviceEnablePeerAccess(g_dest_gpu, 0),
			"MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__)) exit(1);

		// Set device
		if (util::B40CPerror(cudaSetDevice(g_dest_gpu),
			"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
		printf("Enabling peer access to GPU %d from GPU %d\n", g_dest_gpu, g_src_gpu);
		if (util::B40CPerror(cudaDeviceEnablePeerAccess(g_src_gpu, 0),
			"MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__)) exit(1);

	} else {

		if (g_from_host) {
			printf("From pinned host memory.\n");
		}

		// Put current device as both src and dest
		if (util::B40CPerror(cudaGetDevice(&g_src_gpu),
			"MultiGpuBfsEnactor cudaGetDevice failed", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaGetDevice(&g_dest_gpu),
			"MultiGpuBfsEnactor cudaGetDevice failed", __FILE__, __LINE__)) exit(1);
	}
   	TestCopy(g_num_elements);

	return 0;
}



