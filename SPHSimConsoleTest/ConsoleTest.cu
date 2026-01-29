
#include <stdio.h>
#include <memory.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>


typedef unsigned int uint;

// Removed texture references for CUDA 12+ compatibility
// Using direct memory access instead

__global__ void testKernel(float* a, float* b)
{
	// standard multiplication is as fast as __umul24 on sm_20+
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index > 100) index-=100;

	a[index] = 10;

	__threadfence();

	// Changed from tex1Dfetch to direct memory access
	a[index] = a[index] + 1;

	volatile float x = a[index];
	x = a[index];
	x = a[index];
	x = a[index];


	__threadfence();
	__threadfence_block();
//	__threadfence_system();
	__syncthreads();

	//volatile float bv = a[index+1];
	b[index] = a[index+1];
}

void testKernel()
{
	float* da;
	float* db;
	cudaMalloc((void**)&da, 100*sizeof(float));
	cudaMalloc((void**)&db, 100*sizeof(float));
	cudaMemset(da,0,100*sizeof(float));
	cudaMemset(db,0,100*sizeof(float));

	float* ha;
	float* hb;
	cudaMallocHost((void**)&ha, 100*sizeof(float));
	cudaMallocHost((void**)&hb, 100*sizeof(float));
	memset(ha,0,100*sizeof(float));
	memset(hb,0,100*sizeof(float));

	// Removed texture binding - not needed with direct memory access

	testKernel<<<1,101>>>(da,db);

	// Removed texture unbinding

	cudaMemcpy(ha,da,100*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(hb,db,100*sizeof(float),cudaMemcpyDeviceToHost);

	return;
}


