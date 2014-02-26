#include "transpose.h"

#define BLOCK_DIM 16

__global__ void transposeKernel(float* dev_in, float* dev_out, int xRes_in, int yRes_in)
{
	__shared__ float tile[BLOCK_DIM][BLOCK_DIM+1];

	int xIndex = blockIdx.x*BLOCK_DIM + threadIdx.x;
	int yIndex = blockIdx.y*BLOCK_DIM + threadIdx.y;

	//Read in
	if((xIndex < xRes_in) && (yIndex < yRes_in))
	{
		tile[threadIdx.y][threadIdx.x] = dev_in[yIndex * xRes_in + xRes_in];
	}

	__syncthreads();

	//Transpose
	xIndex = blockIdx.y*BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x*BLOCK_DIM + threadIdx.y;

	//Write out
	if((xIndex < yRes_in) && (yIndex < xRes_in))//Ideally this will never be true, but in case matrix dim is not multiple of BLOCK_DIM
	{
		dev_out[yIndex*yRes_in + xIndex] = tile[threadIdx.x][threadIdx.y];
	}

}


__host__ void transpose(float* dev_in, float* dev_out, int xRes_in, int yRes_in)
{

	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks((int)ceil(float(xRes_in)/float(BLOCK_DIM)),
				(int)ceil(float(yRes_in)/float(BLOCK_DIM)));

	transposeKernel<<<threads,blocks>>>(dev_in, dev_out, xRes_in, yRes_in);

}