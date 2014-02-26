#include "array_ops.h"


__global__ void multiplyKernel(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	if(index < size)
	{
		dev_c[index] = dev_a[index]*dev_b[index];
	}
}


__host__ void floatArrayMultiplyCuda(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int tileSize = 128;

	dim3 threads(tileSize);
	dim3 blocks((int)ceil(float(size)/float(tileSize)));

	multiplyKernel<<<threads,blocks>>>(dev_a, dev_b, dev_c, size);

}


__global__ void addKernel(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	if(index < size)
	{
		dev_c[index] = dev_a[index] + dev_b[index];
	}
}


__host__ void floatArrayAdditionCuda(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int tileSize = 128;

	dim3 threads(tileSize);
	dim3 blocks((int)ceil(float(size)/float(tileSize)));

	addKernel<<<threads,blocks>>>(dev_a, dev_b, dev_c, size);

}

__global__ void subtractKernel(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	if(index < size)
	{
		dev_c[index] = dev_a[index] - dev_b[index];
	}
}


__host__ void floatArraySubtractionCuda(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int tileSize = 128;

	dim3 threads(tileSize);
	dim3 blocks((int)ceil(float(size)/float(tileSize)));

	subtractKernel<<<threads,blocks>>>(dev_a, dev_b, dev_c, size);
}


__global__ void divisionKernel(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	if(index < size)
	{
		dev_c[index] = dev_a[index] / dev_b[index];
	}
}


__host__ void floatArrayDivisionCuda(float* dev_a, float* dev_b, float* dev_c, int size)
{
	int tileSize = 128;

	dim3 threads(tileSize);
	dim3 blocks((int)ceil(float(size)/float(tileSize)));

	subtractKernel<<<threads,blocks>>>(dev_a, dev_b, dev_c, size);

}