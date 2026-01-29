#include "UniformGrid.cuh"
#include "CudaUtils.cuh"
#include "cutil.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <utility>


#ifdef SPHSIMLIB_USE_B40C_SORT
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/ping_pong_storage.cuh>
#endif

#ifdef SPHSIMLIB_USE_CUDA_B40C_SORT
#include <thrust/detail/backend/cuda/detail/b40c/radixsort_api.h>
using namespace thrust::detail::backend::cuda::detail::b40c_thrust;
#endif

#ifdef SPHSIMLIB_USE_CUB_SORT
#include <cub/device/device_radix_sort.cuh>
#endif

using namespace std;

// Round up to the next power of 2
static inline uint nextPow2(uint v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

__device__ __constant__	GridParams	cGridParams;

#include "K_UniformGrid_Hash.inl"

using namespace SimLib;

UniformGrid::UniformGrid(SimLib::SimCudaAllocator* simCudaAllocator, SimLib::SimCudaHelper	*simCudaHelper)
	: mSimCudaAllocator(simCudaAllocator)
	, mSimCudaHelper(simCudaHelper)
	, mAlloced(false)
{
	mGPUTimer = new ocu::GPUTimer();

	mGridParticleBuffers = new BufferManager<UniformGridBuffers>(mSimCudaAllocator);
	mGridCellBuffers = new BufferManager<UniformGridBuffers>(mSimCudaAllocator);

	mGridParticleBuffers->SetBuffer(SortHashes,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));
	mGridParticleBuffers->SetBuffer(SortIndexes,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));
	mGridCellBuffers->SetBuffer(CellIndexesStart,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));
	mGridCellBuffers->SetBuffer(CellIndexesStop,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(uint)));

}

UniformGrid::~UniformGrid()
{
	Free();

	delete mGPUTimer; mGPUTimer = NULL;
	delete mGridCellBuffers; mGridCellBuffers = NULL;
	delete mGridParticleBuffers; mGridParticleBuffers = NULL;
}

void UniformGrid::Alloc(uint numParticles, float cellWorldSize, float gridWorldSize)
{
	if(mAlloced)
		Free();

	CalculateGridParameters(cellWorldSize, gridWorldSize);

	mNumParticles = numParticles;

#ifdef SPHSIMLIB_USE_MORTON_HASH
	// Morton codes use 30 bits (10 bits per dimension)
	// Number of unique Morton codes can be up to 2^30
	uint max_res = (uint)max(dGridParams.grid_res.x, max(dGridParams.grid_res.y, dGridParams.grid_res.z));
	mSortBitsPrecision = 3 * (uint)ceil(log2((double)max_res));
	if (mSortBitsPrecision > 30) mSortBitsPrecision = 30;

	// With Morton codes, the number of possible hash values is based on the max dimension
	mNumCells = 1 << mSortBitsPrecision;
#else
	// only need X bits precision for the radix sort.. (256^3 volume ==> 24 bits precision)
	mSortBitsPrecision = (uint)ceil(log2(dGridParams.grid_res.x*dGridParams.grid_res.y*dGridParams.grid_res.z));
	//	assert(mSortBitsPrecision => 4 && mSortBitsPrecision <= 32);

	// number of cells is given by the resolution (~how coarse the grid of the world is)
	mNumCells = (int)ceil(dGridParams.grid_res.x*dGridParams.grid_res.y*dGridParams.grid_res.z);
#endif

	// Allocate grid buffers
	mGridParticleBuffers->AllocBuffers(mNumParticles);
	mGridCellBuffers->AllocBuffers(mNumCells);

	// Allocate the radix sorter
#ifdef SPHSIMLIB_USE_THRUST_SORT
	mThrustKeys = new thrust::device_ptr<uint>(mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>());
	mThrustVals = new thrust::device_ptr<uint>(mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>());
#endif

#ifdef SPHSIMLIB_USE_B40C_SORT
	m_b40c_sorting_enactor = new b40c::radix_sort::Enactor();
	((b40c::radix_sort::Enactor*)m_b40c_sorting_enactor)->ENACTOR_DEBUG = false;

	m_b40c_storage = new b40c::util::PingPongStorage<unsigned int,unsigned int>(mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>(), mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>());	
#endif
#ifdef SPHSIMLIB_USE_CUDA_B40C_SORT
	m_b40c_sorting_enactor = new RadixSortingEnactor<unsigned int, unsigned int>(mNumParticles);
	m_b40c_storage = new RadixSortStorage<unsigned int, unsigned int>(mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>(), mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>());	
#endif

#ifdef SPHSIMLIB_USE_CUDPP_SORT
	// Create the CUDPP radix sort
	CUDPPConfiguration sortConfig;
	sortConfig.algorithm = CUDPP_SORT_RADIX;
	sortConfig.datatype = CUDPP_UINT;
	sortConfig.op = CUDPP_ADD;
	sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cudppPlan(&m_sortHandle, sortConfig, mNumParticles, 1, 0);
#endif

#ifdef SPHSIMLIB_USE_CUB_SORT
	// Allocate alternate buffers for CUB double buffer
	CUDA_SAFE_CALL(mSimCudaAllocator->Allocate((void**)&d_cub_alt_keys, mNumParticles * sizeof(uint)));
	CUDA_SAFE_CALL(mSimCudaAllocator->Allocate((void**)&d_cub_alt_values, mNumParticles * sizeof(uint)));

	// Create double buffer on heap (stored via opaque pointer)
	cub::DoubleBuffer<uint>* keys_buffer = new cub::DoubleBuffer<uint>(
		mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>(),
		d_cub_alt_keys);
	cub::DoubleBuffer<uint>* values_buffer = new cub::DoubleBuffer<uint>(
		mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>(),
		d_cub_alt_values);
	d_cub_double_buffer = new std::pair<cub::DoubleBuffer<uint>*, cub::DoubleBuffer<uint>*>(keys_buffer, values_buffer);

	// Query temp storage size
	d_cub_temp_storage = nullptr;
	d_cub_temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(
		d_cub_temp_storage, d_cub_temp_storage_bytes,
		*keys_buffer, *values_buffer,
		mNumParticles, 0, mSortBitsPrecision);

	// Allocate temp storage
	CUDA_SAFE_CALL(mSimCudaAllocator->Allocate(&d_cub_temp_storage, d_cub_temp_storage_bytes));
#endif

	//Copy the grid parameters to the GPU
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cGridParams, &dGridParams, sizeof(GridParams) ) );

	mAlloced = true;
}

void UniformGrid::Free()	
{
	if(!mAlloced)
		return;

#ifdef SPHSIMLIB_USE_THRUST_SORT
	delete mThrustKeys; mThrustKeys = NULL;
	delete mThrustVals; mThrustVals = NULL;
#endif
#ifdef SPHSIMLIB_USE_B40C_SORT
	delete m_b40c_sorting_enactor;
	delete m_b40c_storage;
#endif
#ifdef SPHSIMLIB_USE_CUDA_B40C_SORT
	RadixSortStorage<unsigned int, unsigned int>* storage = (RadixSortStorage<unsigned int, unsigned int>*)m_b40c_storage;
	storage->CleanupTempStorage();
	delete m_b40c_sorting_enactor;
	delete m_b40c_storage;
#endif
#ifdef SPHSIMLIB_USE_CUDPP_SORT
	cudppDestroyPlan(m_sortHandle);	m_sortHandle=NULL;
#endif

#ifdef SPHSIMLIB_USE_CUB_SORT
	if (d_cub_temp_storage) {
		CUDA_SAFE_CALL(mSimCudaAllocator->Free(&d_cub_temp_storage));
		d_cub_temp_storage = nullptr;
	}
	if (d_cub_alt_keys) {
		CUDA_SAFE_CALL(mSimCudaAllocator->Free((void**)&d_cub_alt_keys));
		d_cub_alt_keys = nullptr;
	}
	if (d_cub_alt_values) {
		CUDA_SAFE_CALL(mSimCudaAllocator->Free((void**)&d_cub_alt_values));
		d_cub_alt_values = nullptr;
	}
	if (d_cub_double_buffer) {
		auto* buffers = static_cast<std::pair<cub::DoubleBuffer<uint>*, cub::DoubleBuffer<uint>*>*>(d_cub_double_buffer);
		delete buffers->first;
		delete buffers->second;
		delete buffers;
		d_cub_double_buffer = nullptr;
	}
#endif

	mGridParticleBuffers->FreeBuffers();
	mGridCellBuffers->FreeBuffers();

	mAlloced = false;
}

GridData UniformGrid::GetGridData()
{
	GridData gridData;
#if defined SPHSIMLIB_USE_CUB_SORT
	// CUB uses double buffers - read from Current() which points to sorted output
	auto* buffers = static_cast<std::pair<cub::DoubleBuffer<uint>*, cub::DoubleBuffer<uint>*>*>(d_cub_double_buffer);
	gridData.sort_hashes = buffers->first->Current();
	gridData.sort_indexes = buffers->second->Current();
#elif defined SPHSIMLIB_USE_CUDA_B40C_SORT
	// if using b40c the results of the sort "ping-pong" between two buffers
	// we select the "current" results using the pingpongstorage selector.
	RadixSortStorage<unsigned int, unsigned int>* storage = (RadixSortStorage<unsigned int, unsigned int>*)m_b40c_storage;
	if(storage->d_from_alt_storage)
	{
		gridData.sort_hashes = storage->d_alt_keys;
		gridData.sort_indexes = storage->d_alt_values;
	}
	else
	{
		gridData.sort_hashes = storage->d_keys;
		gridData.sort_indexes = storage->d_values;
	}
#elif defined SPHSIMLIB_USE_B40C_SORT
	// if using b40c the results of the sort "ping-pong" between two buffers
	// we select the "current" results using the pingpongstorage selector.
	b40c::util::PingPongStorage<unsigned int,unsigned int>* storage = ((b40c::util::PingPongStorage<unsigned int,unsigned int>*)m_b40c_storage);
	gridData.sort_hashes = storage->d_keys[storage->selector];
	gridData.sort_indexes = storage->d_values[storage->selector];
#else
	gridData.sort_hashes = mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>();
	gridData.sort_indexes = mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>();
#endif
	gridData.cell_indexes_start = mGridCellBuffers->Get(CellIndexesStart)->GetPtr<uint>();
	gridData.cell_indexes_end = mGridCellBuffers->Get(CellIndexesStop)->GetPtr<uint>();
	return gridData;
}

void UniformGrid::Clear()
{
// 	mGridCellBuffers->ClearBuffers();
// 	mGridParticleBuffers->ClearBuffers();
}

void UniformGrid::CalculateGridParameters(float cellWorldSize, float gridWorldSize)
{
	// GRID SETUP

	// Ideal grid "cell" size (gs) = 2 * smoothing length	.. then we can use 8 cell checker
	// however ... atm we use particles 27 cell checker, so cell size must be equal to smoothing length
	dGridParams.grid_min = make_float3(0, 0, 0);
	dGridParams.grid_max = dGridParams.grid_min + (float)gridWorldSize;

	dGridParams.grid_size = make_float3(
		dGridParams.grid_max.x-dGridParams.grid_min.x,
		dGridParams.grid_max.y-dGridParams.grid_min.y,
		dGridParams.grid_max.z-dGridParams.grid_min.z);

	dGridParams.grid_res = make_float3(
		ceil(dGridParams.grid_size.x / cellWorldSize),
		ceil(dGridParams.grid_size.y / cellWorldSize),
		ceil(dGridParams.grid_size.z / cellWorldSize));

	// Adjust grid size to multiple of cell size
	dGridParams.grid_size.x = dGridParams.grid_res.x * cellWorldSize;
	dGridParams.grid_size.y = dGridParams.grid_res.y * cellWorldSize;
	dGridParams.grid_size.z = dGridParams.grid_res.z * cellWorldSize;

	dGridParams.grid_delta.x = dGridParams.grid_res.x / dGridParams.grid_size.x;
	dGridParams.grid_delta.y = dGridParams.grid_res.y / dGridParams.grid_size.y;
	dGridParams.grid_delta.z = dGridParams.grid_res.z / dGridParams.grid_size.z;
}


float UniformGrid::Hash(bool doTiming, float_vec* dParticlePositions, uint numParticles)
{
//	assert(mNumParticles == numParticles);

	// clear old hash values
	mGridParticleBuffers->Get(SortHashes)->Memset(0);

	int threadsPerBlock = 256;

	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);

	GridData dGridData = GetGridData();

	if(doTiming)
	{
		mGPUTimer->start();
	}

	// hash each particle according to spatial position (cell in grid volume)
	K_Grid_Hash<<< numBlocks, numThreads>>> (
		mNumParticles,
		dParticlePositions, 
		dGridData
		);	

	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

float UniformGrid::Sort(bool doTiming)
{
	if(doTiming)
	{
		mGPUTimer->start();
	}

#ifdef SPHSIMLIB_USE_THRUST_SORT
	thrust::sort_by_key(*mThrustKeys, (*mThrustKeys)+mNumParticles, *mThrustVals);	
#endif
#ifdef SPHSIMLIB_USE_B40C_SORT
	
	b40c::util::PingPongStorage<unsigned int,unsigned int>* storage = ((b40c::util::PingPongStorage<unsigned int,unsigned int>*)m_b40c_storage);
	b40c::radix_sort::Enactor* enactor = ((b40c::radix_sort::Enactor*)m_b40c_sorting_enactor);
	enactor->Sort<0, 24, b40c::radix_sort::SMALL_SIZE>(*storage, mNumParticles);
	
#endif
#ifdef SPHSIMLIB_USE_CUDA_B40C_SORT
	RadixSortStorage<unsigned int,unsigned int>* storage = ((RadixSortStorage<unsigned int,unsigned int>*)m_b40c_storage);
	RadixSortingEnactor<unsigned int,unsigned int>* enactor = ((RadixSortingEnactor<unsigned int,unsigned int>*)m_b40c_sorting_enactor);
	enactor->EnactSort(*storage);
#endif

#ifdef SPHSIMLIB_USE_CUDPP_SORT
	cudppSort(
		m_sortHandle,
		mGridParticleBuffers->Get(SortHashes)->GetPtr<uint>(),
		mGridParticleBuffers->Get(SortIndexes)->GetPtr<uint>(),
		mSortBitsPrecision,
		mNumParticles);
#endif

#ifdef SPHSIMLIB_USE_CUB_SORT
	auto* buffers = static_cast<std::pair<cub::DoubleBuffer<uint>*, cub::DoubleBuffer<uint>*>*>(d_cub_double_buffer);
	cub::DeviceRadixSort::SortPairs(
		d_cub_temp_storage, d_cub_temp_storage_bytes,
		*buffers->first, *buffers->second,
		mNumParticles, 0, mSortBitsPrecision);
#endif

	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}
