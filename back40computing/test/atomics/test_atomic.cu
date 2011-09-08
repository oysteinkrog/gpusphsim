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
 * Simple test driver program for atomics.
 ******************************************************************************/

#include <stdio.h> 

#include "b40c_test_util.h"

#include "b40c/util/cuda_properties.cuh"
#include "b40c/util/error_utils.cuh"

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
int 	g_num_elements 					= 1024;
int 	g_log_stride					= -2;
bool 	g_stream						= false;
bool 	g_local							= false;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_atomic [--device=<device index>] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] "
			"[log-stride=<log2(access stride)>] [--stream] [--local]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the atomic pass <num-iterations> times.\n");
	printf("\t\t\tDefault = %d\n", g_iterations);
	printf("\n");
	printf("\t--n\tThe number of bytes to comprise the sample problem\n");
	printf("\t\t\tDefault = %lu\n", (unsigned long) g_num_elements);
	printf("\n");
	printf("\t--log-stride\tThe log of the stride in threads between threads \n");
	printf("\t\t\tactively updating the shared counter. -1 for no atomics. Default = %lu\n", (unsigned long) g_log_stride);
	printf("\n");
	printf("\t--stream\tWhether or not threads also stream 32-bit words \n");
	printf("\t\t\tthrough global memory concurrently as well (to simulate)\n");
	printf("\n");
	printf("\t--local\tWhether or not to perform atomics on local smem \n");
	printf("\t\t\tcounters versus a global gmem counter\n");
	printf("\n");
}


/******************************************************************************
 * Kernels
 ******************************************************************************/

/**
 * Updates global counter
 */
__global__ void GlobalAtomicKernel(
	int *d_counter,
	int *d_in,
	int *d_out,
	int stride,
	int tiles_per_cta,
	int extra_tiles)
{
	int data;
	int tid = (blockIdx.x * blockDim.x * tiles_per_cta) + threadIdx.x;
	int mask = stride - 1;

	if (blockIdx.x < extra_tiles) {

		// We get an extra tile
		tiles_per_cta++;
		tid += blockIdx.x * blockDim.x;

	} else if (extra_tiles > 0) {

		// We don't get an extra tile, but others did
		tid += blockDim.x * extra_tiles;
	}

	// Iterate over tiles
	for (int i = 0; i < tiles_per_cta; i++) {

		if (d_in) {
			data = d_in[tid];
		}

		if ((tid & mask) == mask) {
			if (stride > 0) {
				// access shared counter
				atomicAdd(d_counter, 1);
			}
		}

		if (d_out) {
			d_out[tid] = data;
		}

		tid += blockDim.x;
	}
}


/**
 * Updates local counter
 */
__global__ void LocalAtomicKernel(
	int *d_in,
	int *d_out,
	int stride,
	int tiles_per_cta,
	int extra_tiles)
{
	__shared__ int s_counter;

	int data;
	int tid = (blockIdx.x * blockDim.x * tiles_per_cta) + threadIdx.x;
	int mask = stride - 1;

	if (blockIdx.x < extra_tiles) {

		// We get an extra tile
		tiles_per_cta++;
		tid += blockIdx.x * blockDim.x;

	} else if (extra_tiles > 0) {

		// We don't get an extra tile, but others did
		tid += blockDim.x * extra_tiles;
	}

	// Iterate over tiles
	for (int i = 0; i < tiles_per_cta; i++) {

		if (d_in) {
			data = d_in[tid];
		}

		if ((tid & mask) == mask) {
			if (stride > 0) {

#if __B40C_CUDA_ARCH__ >= 120
				// access shared counter
				atomicAdd(&s_counter, 1);
#endif
			}
		}

		if (d_out) {
			d_out[tid] = data;
		}

		tid += blockDim.x;
	}
}


/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * Main
 */
int main(int argc, char** argv)
{
    const int LOG_CTA_THREADS = 8;
    const int CTA_THREADS = 1 << LOG_CTA_THREADS;
    const int MAX_CTAS = 1024 * 16;


    CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Check command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    g_stream = args.CheckCmdLineFlag("stream");
    g_local = args.CheckCmdLineFlag("local");
    args.GetCmdLineArgument("log-stride", g_log_stride);
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", g_num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);

	// Allocate problem
    int *d_counter = NULL;
    int *d_in = NULL;
    int *d_out = NULL;

	if (util::B40CPerror(cudaMalloc((void**) &d_counter, sizeof(int) * 1),
		"cudaMalloc d_counter failed: ", __FILE__, __LINE__)) exit(1);
    if (g_stream) {
    	if (util::B40CPerror(cudaMalloc((void**) &d_in, sizeof(int) * g_num_elements),
    		"cudaMalloc d_in failed: ", __FILE__, __LINE__)) exit(1);
    	if (util::B40CPerror(cudaMalloc((void**) &d_out, sizeof(int) * g_num_elements),
    		"cudaMalloc d_out failed: ", __FILE__, __LINE__)) exit(1);
    }

    // Compute kernel params
    util::CudaProperties cuda_props;
    if (g_max_ctas <= 0) {
        g_max_ctas = MAX_CTAS;
    }

    int tiles = (g_num_elements + CTA_THREADS - 1) / CTA_THREADS;
    g_num_elements = tiles * CTA_THREADS;

    int grid_size = (tiles > g_max_ctas) ?
    	g_max_ctas :
    	tiles;

    int tiles_per_cta = tiles / grid_size;
    int extra_tiles = tiles - (tiles_per_cta * grid_size);

	printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
		cuda_props.device_sm_version,
		cuda_props.kernel_ptx_version);
    printf("%d iterations, %d elements, %d tile size, %d tiles, %d CTAs, %d tiles/CTA, %d extra tiles\n",
    	g_iterations, g_num_elements, CTA_THREADS, tiles, grid_size, tiles_per_cta, extra_tiles);
    printf("\n");
    printf("Log-Stride, Stride, Elapsed time (ms), 10^9 atomics/sec");
    if (d_in) printf(", 10^9 bytes/sec");
    printf("\n");

    int min_log_stride = 0;
    int max_log_stride = 18;
    if (g_log_stride > -2) {
    	min_log_stride = g_log_stride;
    	max_log_stride = g_log_stride;
    }

    for (
    	int log_stride = min_log_stride;
    	log_stride <= max_log_stride;
    	log_stride++)
    {
    	int stride = (log_stride >= 0) ?
    		1 << log_stride :
    		0;

		// Perform the timed number of iterations
		GpuTimer timer;

		timer.Start();
		for (int i = 0; i < g_iterations; i++) {

			if (g_local) {
				LocalAtomicKernel<<<grid_size, CTA_THREADS>>>(
					d_in,
					d_out,
					stride,
					tiles_per_cta,
					extra_tiles);
			} else {
				GlobalAtomicKernel<<<grid_size, CTA_THREADS>>>(
					d_counter,
					d_in,
					d_out,
					stride,
					tiles_per_cta,
					extra_tiles);
			}
		}
		timer.Stop();

		float avg_millis = timer.ElapsedMillis() / g_iterations;
		float ops = float(g_num_elements) / stride;

		printf("%d, %d, %.5f, %.5f",
			log_stride,
			stride,
			avg_millis,
			ops / avg_millis / 1000.0 / 1000.0);
		if (d_in) {
			unsigned long long bytes = g_num_elements;
			bytes *= sizeof(int) * 2;
			printf(", %.5f",
				float(bytes) / avg_millis / 1000.0 / 1000.0);
		}
		printf("\n");
		fflush(stdout);
    }

	// Cleanup
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_counter) cudaFree(d_counter);

    cudaThreadSynchronize();


	return 0;
}



