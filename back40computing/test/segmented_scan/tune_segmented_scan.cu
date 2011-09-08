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
 * Tuning tool for establishing optimal segmented scan granularity configuration types
 ******************************************************************************/

#include <stdio.h> 

// SegmentedScan includes
#include <b40c/util/arch_dispatch.cuh>
#include <b40c/segmented_scan/problem_type.cuh>
#include <b40c/segmented_scan/problem_config.cuh>
#include <b40c/segmented_scan/enactor.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/numeric_traits.cuh>
#include <b40c/util/parameter_generation.cuh>

// Test utils
#include "b40c_test_util.h"

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals, and utility types
 ******************************************************************************/

#ifndef TUNE_ARCH
	#define TUNE_ARCH (200)
#endif
#ifndef TUNE_SIZE
	#define TUNE_SIZE (4)
#endif

bool g_verbose;
int g_max_ctas = 0;
int g_iterations = 0;


template <typename T>
struct Sum
{
	static __host__ __device__ __forceinline__ T BinaryOp(const T &a, const T &b)
	{
		return a + b;
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return 0;
	}
};

template <typename T>
struct Max
{
	static __host__ __device__ __forceinline__ T BinaryOp(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return 0;
	}
};



/******************************************************************************
 * Utility routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntune_segmented_scan [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>]\n");
	printf("\n");
	printf("\t--v\tDisplays verbose configuration to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the segmented scan operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}

/******************************************************************************
 * Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

/**
 * Enumerated tuning params
 */
enum TuningParam {

	PARAM_BEGIN,

		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,

		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_TILE,

	PARAM_END,

	// Parameters below here are currently not part of the tuning sweep

	// These can be tuned, but we're currently not compelled to
	UNIFORM_GRID_SIZE,
	UNIFORM_SMEM_ALLOCATION,
	OVERSUBSCRIBED_GRID_SIZE,
	READ_MODIFIER,
	WRITE_MODIFIER,
	UPSWEEP_LOG_RAKING_THREADS,
	DOWNSWEEP_LOG_RAKING_THREADS,

	// Derive these from the others above
	UPSWEEP_MAX_CTA_OCCUPANCY,
	DOWNSWEEP_MAX_CTA_OCCUPANCY,
	LOG_SCHEDULE_GRANULARITY,

	// General performance is insensitive to the spine kernel params
	// because it's only a single-CTA: we'll just use reasonable defaults
	SPINE_LOG_THREADS,
	SPINE_LOG_LOAD_VEC_SIZE,
	SPINE_LOG_LOADS_PER_TILE,
	SPINE_LOG_RAKING_THREADS
};


/**
 * Encapsulation structure for
 * 		- Wrapping problem type and storage
 * 		- Providing call-back for parameter-list generation
 */
template <typename T, typename Flag, typename OpType, bool EXCLUSIVE>
class TuneEnactor : public segmented_scan::Enactor
{
public:

	T *d_dest;
	T *d_src;
	Flag *d_flag_src;
	T *h_data;
	T *h_reference;
	Flag *h_flag_data;
	size_t num_elements;

	/**
	 * Ranges for the tuning params
	 */
	template <typename ParamList, int PARAM> struct Ranges;

	// READ_MODIFIER
	template <typename ParamList>
	struct Ranges<ParamList, READ_MODIFIER> {
		enum {
			MIN = util::ld::NONE,
			MAX = ((TUNE_ARCH < 200) || (util::NumericTraits<T>::REPRESENTATION == util::NOT_A_NUMBER)) ? util::ld::NONE : util::ld::CS		// No type modifiers for pre-Fermi or non-builtin types
		};
	};

	// WRITE_MODIFIER
	template <typename ParamList>
	struct Ranges<ParamList, WRITE_MODIFIER> {
		enum {
			MIN = util::st::NONE,
			MAX = ((TUNE_ARCH < 200) || (util::NumericTraits<T>::REPRESENTATION == util::NOT_A_NUMBER)) ? util::st::NONE : util::st::CS		// No type modifiers for pre-Fermi or non-builtin types
		};
	};

	// UNIFORM_SMEM_ALLOCATION
	template <typename ParamList>
	struct Ranges<ParamList, UNIFORM_SMEM_ALLOCATION> {
		enum {
			MIN = 0,
			MAX = 1
		};
	};

	// UNIFORM_GRID_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, UNIFORM_GRID_SIZE> {
		enum {
			MIN = 0,
			MAX = 1
		};
	};

	// OVERSUBSCRIBED_GRID_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, OVERSUBSCRIBED_GRID_SIZE> {
		enum {
			MIN = 0,
			MAX = 1
		};
	};

	// UPSWEEP_LOG_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, UPSWEEP_LOG_THREADS> {
		enum {
			MIN = B40C_LOG_WARP_THREADS(TUNE_ARCH),
			MAX = B40C_LOG_CTA_THREADS(TUNE_ARCH)
		};
	};

	// UPSWEEP_LOG_LOAD_VEC_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, UPSWEEP_LOG_LOAD_VEC_SIZE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// UPSWEEP_LOG_LOADS_PER_TILE
	template <typename ParamList>
	struct Ranges<ParamList, UPSWEEP_LOG_LOADS_PER_TILE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// UPSWEEP_LOG_RAKING_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, UPSWEEP_LOG_RAKING_THREADS> {
		enum {
			MIN = B40C_LOG_WARP_THREADS(TUNE_ARCH),
			MAX = util::Access<ParamList, UPSWEEP_LOG_THREADS>::VALUE
		};
	};

	// DOWNSWEEP_LOG_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, DOWNSWEEP_LOG_THREADS> {
		enum {
			MIN = B40C_LOG_WARP_THREADS(TUNE_ARCH),
			MAX = B40C_LOG_CTA_THREADS(TUNE_ARCH)
		};
	};

	// DOWNSWEEP_LOG_LOAD_VEC_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, DOWNSWEEP_LOG_LOAD_VEC_SIZE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// DOWNSWEEP_LOG_LOADS_PER_TILE
	template <typename ParamList>
	struct Ranges<ParamList, DOWNSWEEP_LOG_LOADS_PER_TILE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// DOWNSWEEP_LOG_RAKING_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, DOWNSWEEP_LOG_RAKING_THREADS> {
		enum {
			MIN = B40C_LOG_WARP_THREADS(TUNE_ARCH),
			MAX = util::Access<ParamList, DOWNSWEEP_LOG_THREADS>::VALUE
		};
	};


	/**
	 * Constructor
	 */
	TuneEnactor(size_t num_elements) :
		segmented_scan::Enactor(), d_dest(NULL), d_src(NULL), d_flag_src(NULL), h_data(NULL), h_reference(NULL), num_elements(num_elements) {}


	/**
	 * Timed segmented scan for applying a specific granularity configuration type
	 */
	template <typename ProblemConfig>
	void TimedSegmentedScan()
	{
		printf("%lu, ", (unsigned long) sizeof(T));
		ProblemConfig::Print();
		fflush(stdout);

		// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
		this->ENACTOR_DEBUG = g_verbose;
		if (this->template Enact<ProblemConfig>(d_dest, d_src, d_flag_src, num_elements, g_max_ctas)) {
			exit(1);
		}
		this->ENACTOR_DEBUG = false;

		// Perform the timed number of iterations

		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);

		double elapsed = 0;
		float duration = 0;
		for (int i = 0; i < g_iterations; i++) {

			// Start cuda timing record
			cudaEventRecord(start_event, 0);

			// Call the segmented scan API routine
			if (this->template Enact<ProblemConfig>(d_dest, d_src, d_flag_src, num_elements, g_max_ctas)) {
				exit(1);
			}

			// End cuda timing record
			cudaEventRecord(stop_event, 0);
			cudaEventSynchronize(stop_event);
			cudaEventElapsedTime(&duration, start_event, stop_event);
			elapsed += (double) duration;

			// Flushes any stdio from the GPU
			cudaThreadSynchronize();
		}

		// Display timing information
		double avg_runtime = elapsed / g_iterations;
		double throughput =  0.0;
		if (avg_runtime > 0.0) throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	    printf(", %f, %f, %f, ",
			avg_runtime, throughput, throughput * ((sizeof(T) * 3) + (sizeof(Flag) * 2)));
	    fflush(stdout);

	    // Clean up events
		cudaEventDestroy(start_event);
		cudaEventDestroy(stop_event);

	    // Copy out data
	    if (util::B40CPerror(cudaMemcpy(h_data, d_dest, sizeof(T) * num_elements, cudaMemcpyDeviceToHost),
			"TimedSegmentedScan cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

	    // Verify solution
		CompareResults<T>(h_data, h_reference, num_elements, true);
		printf("\n");
		fflush(stdout);
	}

	template <typename ProblemConfig, bool VALID>
	struct LaunchValidConfig
	{
		static void Invoke(TuneEnactor *detail)
		{
			detail->TimedSegmentedScan<ProblemConfig>();
		}
	};


	template <typename ProblemConfig>
	struct LaunchValidConfig <ProblemConfig, false>
	{
		static void Invoke(TuneEnactor *detail) {}
	};

	/**
	 * Callback invoked by parameter-list generation
	 */
	template <typename ParamList>
	void Invoke()
	{
		const int C_READ_MODIFIER =
//			util::Access<ParamList, READ_MODIFIER>::VALUE;					// These can be tuned, but we're currently not compelled to
			util::ld::NONE;
		const int C_WRITE_MODIFIER =
//			util::Access<ParamList, WRITE_MODIFIER>::VALUE;					// These can be tuned, but we're currently not compelled to
			util::ld::NONE;
		const int C_UNIFORM_SMEM_ALLOCATION =
//			util::Access<ParamList, UNIFORM_SMEM_ALLOCATION>::VALUE;
			0;
		const int C_UNIFORM_GRID_SIZE =
//			util::Access<ParamList, UNIFORM_GRID_SIZE>::VALUE;
			0;
		const int C_OVERSUBSCRIBED_GRID_SIZE =
//			util::Access<ParamList, OVERSUBSCRIBED_GRID_SIZE>::VALUE;
			0;

		const int C_UPSWEEP_LOG_THREADS =
			util::Access<ParamList, UPSWEEP_LOG_THREADS>::VALUE;
		const int C_UPSWEEP_LOG_LOAD_VEC_SIZE =
			util::Access<ParamList, UPSWEEP_LOG_LOAD_VEC_SIZE>::VALUE;
		const int C_UPSWEEP_LOG_LOADS_PER_TILE =
			util::Access<ParamList, UPSWEEP_LOG_LOADS_PER_TILE>::VALUE;
		const int C_UPSWEEP_LOG_RAKING_THREADS =
//			util::Access<ParamList, UPSWEEP_LOG_RAKING_THREADS>::VALUE;		// These can be tuned, but we're currently not compelled to
			B40C_LOG_WARP_THREADS(TUNE_ARCH);
		const int C_UPSWEEP_MAX_CTA_OCCUPANCY =
//			util::Access<ParamList, UPSWEEP_MAX_CTA_OCCUPANCY>::VALUE;
			B40C_SM_CTAS(TUNE_ARCH);

		const int C_DOWNSWEEP_LOG_THREADS =
			util::Access<ParamList, DOWNSWEEP_LOG_THREADS>::VALUE;
		const int C_DOWNSWEEP_LOG_LOAD_VEC_SIZE =
			util::Access<ParamList, DOWNSWEEP_LOG_LOAD_VEC_SIZE>::VALUE;
		const int C_DOWNSWEEP_LOG_LOADS_PER_TILE =
			util::Access<ParamList, DOWNSWEEP_LOG_LOADS_PER_TILE>::VALUE;
		const int C_DOWNSWEEP_LOG_RAKING_THREADS =
//			util::Access<ParamList, DOWNSWEEP_LOG_RAKING_THREADS>::VALUE;		// These can be tuned, but we're currently not compelled to
			B40C_LOG_WARP_THREADS(TUNE_ARCH);
		const int C_DOWNSWEEP_MAX_CTA_OCCUPANCY =
//			util::Access<ParamList, DOWNSWEEP_MAX_CTA_OCCUPANCY>::VALUE;
			B40C_SM_CTAS(TUNE_ARCH);


		const int C_UPSWEEP_LOG_SCHEDULE_GRANULARITY =
			C_UPSWEEP_LOG_LOADS_PER_TILE +
			C_UPSWEEP_LOG_LOAD_VEC_SIZE +
			C_UPSWEEP_LOG_THREADS;

		const int C_DOWNSWEEP_LOG_SCHEDULE_GRANULARITY =
			C_DOWNSWEEP_LOG_LOADS_PER_TILE +
			C_DOWNSWEEP_LOG_LOAD_VEC_SIZE +
			C_DOWNSWEEP_LOG_THREADS;

		// TODO: figure out if we should use min here instead
		const int C_LOG_SCHEDULE_GRANULARITY = B40C_MAX(
			C_UPSWEEP_LOG_SCHEDULE_GRANULARITY,
			C_DOWNSWEEP_LOG_SCHEDULE_GRANULARITY);

		// General performance is insensitive to spine config it's only a single-CTA:
		// simply use reasonable defaults
		const int C_SPINE_LOG_THREADS = 8;
		const int C_SPINE_LOG_LOAD_VEC_SIZE = 0;
		const int C_SPINE_LOG_LOADS_PER_TILE = 1;
		const int C_SPINE_LOG_RAKING_THREADS = B40C_LOG_WARP_THREADS(TUNE_ARCH);
		
		// Establish the problem type
		typedef segmented_scan::ProblemType<
			T,
			Flag,
			size_t,
			EXCLUSIVE,
			OpType::BinaryOp,
			OpType::Identity> ProblemType;

		// Establish the granularity configuration type
		typedef segmented_scan::ProblemConfig <
			ProblemType,
			TUNE_ARCH,
			(util::io::ld::CacheModifier) C_READ_MODIFIER,
			(util::io::st::CacheModifier) C_WRITE_MODIFIER,
			C_UNIFORM_SMEM_ALLOCATION,
			C_UNIFORM_GRID_SIZE,
			C_OVERSUBSCRIBED_GRID_SIZE,
			C_LOG_SCHEDULE_GRANULARITY,

			C_UPSWEEP_MAX_CTA_OCCUPANCY,
			C_UPSWEEP_LOG_THREADS,
			C_UPSWEEP_LOG_LOAD_VEC_SIZE,
			C_UPSWEEP_LOG_LOADS_PER_TILE,
			C_UPSWEEP_LOG_RAKING_THREADS,

			C_SPINE_LOG_THREADS, 
			C_SPINE_LOG_LOAD_VEC_SIZE, 
			C_SPINE_LOG_LOADS_PER_TILE, 
			C_SPINE_LOG_RAKING_THREADS,

			C_DOWNSWEEP_MAX_CTA_OCCUPANCY,
			C_DOWNSWEEP_LOG_THREADS,
			C_DOWNSWEEP_LOG_LOAD_VEC_SIZE,
			C_DOWNSWEEP_LOG_LOADS_PER_TILE,
			C_DOWNSWEEP_LOG_RAKING_THREADS> ProblemConfig;

		LaunchValidConfig<ProblemConfig, ProblemConfig::VALID>::Invoke(this);
	}
};


/**
 * Creates an example segmented scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<typename T, typename Flag, typename OpType>
void TestSegmentedScan(size_t num_elements)
{
	const bool EXCLUSIVE = true;

	// Allocate storage and enactor
	typedef TuneEnactor<T, Flag, OpType, EXCLUSIVE> Detail;
	Detail detail(num_elements);

	if (util::B40CPerror(cudaMalloc((void**) &detail.d_src, sizeof(T) * num_elements),
		"TimedSegmentedScan cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &detail.d_dest, sizeof(T) * num_elements),
		"TimedSegmentedScan cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &detail.d_flag_src, sizeof(Flag) * num_elements),
		"TimedSegmentedScan cudaMalloc d_flag_src failed: ", __FILE__, __LINE__)) exit(1);

	if ((detail.h_data = (T*) malloc(sizeof(T) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}
	if ((detail.h_reference = (T*) malloc(sizeof(T) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}
	if ((detail.h_flag_data = (Flag*) malloc(sizeof(Flag) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (size_t i = 0; i < num_elements; ++i) {
//		util::RandomBits<T>(detail.h_data[i], 0);
//		util::RandomBits<Flag>(detail.h_flag_data[i], 0);
		detail.h_data[i] = 1;
		detail.h_flag_data[i] = (i % 11) == 0;
	}

	for (size_t i = 0; i < num_elements; ++i) {
		if (EXCLUSIVE)
		{
			detail.h_reference[i] = ((i == 0) || (detail.h_flag_data[i])) ?
				OpType::Identity() :
				OpType::BinaryOp(detail.h_reference[i - 1], detail.h_data[i - 1]);
		} else {
			detail.h_reference[i] = ((i == 0) || (detail.h_flag_data[i])) ?
				detail.h_data[i] :
				OpType::BinaryOp(detail.h_reference[i - 1], detail.h_data[i]);
		}
	}

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(detail.d_src, detail.h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedSegmentedScan cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);
	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(detail.d_flag_src, detail.h_flag_data, sizeof(Flag) * num_elements, cudaMemcpyHostToDevice),
		"TimedSegmentedScan cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	// Run the timing tests
	util::ParamListSweep<
		Detail,
		PARAM_BEGIN + 1,
		PARAM_END,
		Detail::template Ranges>::template Invoke<util::EmptyTuple>(detail);

	// Free allocated memory
	if (detail.d_src) cudaFree(detail.d_src);
	if (detail.d_dest) cudaFree(detail.d_dest);
	if (detail.d_flag_src) cudaFree(detail.d_flag_src);

	// Free our allocated host memory
	if (detail.h_data) free(detail.h_data);
	if (detail.h_reference) free(detail.h_reference);
	if (detail.h_flag_data) free(detail.h_flag_data);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	// Initialize commandline args and device
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Seed random number generator
	srand(0);				// presently deterministic
	//srand(time(NULL));

	// Use 32-bit integer for array indexing
	typedef int SizeT;
	SizeT num_elements = 1024;

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	util::CudaProperties cuda_props;

	printf("Test SegmentedScan: %d iterations, %lu elements", g_iterations, (unsigned long) num_elements);
	printf("\nCodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n\n",
		cuda_props.device_sm_version, cuda_props.kernel_ptx_version);

	printf("sizeof(T), READ_MODIFIER, WRITE_MODIFIER, UNIFORM_SMEM_ALLOCATION, UNIFORM_GRID_SIZE, OVERSUBSCRIBED_GRID_SIZE, LOG_SCHEDULE_GRANULARITY, "
		"UPSWEEP_MAX_CTA_OCCUPANCY, UPSWEEP_LOG_THREADS, UPSWEEP_LOG_LOAD_VEC_SIZE, UPSWEEP_LOG_LOADS_PER_TILE, UPSWEEP_LOG_RAKING_THREADS, "
		"SPINE_LOG_THREADS, SPINE_LOG_LOAD_VEC_SIZE, SPINE_LOG_LOADS_PER_TILE, SPINE_LOG_RAKING_THREADS, "
		"DOWNSWEEP_MAX_CTA_OCCUPANCY, DOWNSWEEP_LOG_THREADS, DOWNSWEEP_LOG_LOAD_VEC_SIZE, DOWNSWEEP_LOG_LOADS_PER_TILE, DOWNSWEEP_LOG_RAKING_THREADS, "
		"elapsed time (ms), throughput (10^9 items/s), bandwidth (10^9 B/s), Correctness\n");

	typedef unsigned char Flag;

	// Execute test(s)
#if TUNE_SIZE == 1
	typedef unsigned char T;
	TestSegmentedScan<T, Flag, Sum<T> >(num_elements * 4);
#elif TUNE_SIZE == 2
	typedef unsigned short T;
	TestSegmentedScan<T, Flag, Sum<T> >(num_elements * 2);
#elif TUNE_SIZE == 4
	typedef unsigned int T;
	TestSegmentedScan<T, Flag, Sum<T> >(num_elements);
#elif TUNE_SIZE == 8
	typedef unsigned long long T;
	TestSegmentedScan<T, Flag, Sum<T> >(num_elements / 2);
#endif

	return 0;

}



