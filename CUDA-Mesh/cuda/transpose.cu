#include "transpose.h"

#define BLOCK_DIM 16

__global__ void transposeKernel(float* dev_in, float* dev_out, int xRes_in, int yRes_in)
{
	__shared__ float tile[BLOCK_DIM][BLOCK_DIM+1];

	int bx = blockIdx.x * BLOCK_DIM;
	int by = blockIdx.y * BLOCK_DIM;

	int i = by + threadIdx.y;//Row
	int j = bx + threadIdx.x;//Column
	int ti = bx + threadIdx.y;//Transposed coallesed writeback location row
	int tj = by + threadIdx.x;//Transposed coallesed writeback location column

	//Read in
	if(i < yRes_in && j < xRes_in)
		tile[threadIdx.x][threadIdx.y] = dev_in[i * xRes_in + j];


	__syncthreads();

	if(tj < xRes_in && ti < yRes_in)
		dev_out[ti * yRes_in + tj] = tile[threadIdx.y][threadIdx.x];
}


__host__ void transpose(float* dev_in, float* dev_out, int xRes_in, int yRes_in)
{

	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks((int)ceil(float(xRes_in)/float(BLOCK_DIM)),
				(int)ceil(float(yRes_in)/float(BLOCK_DIM)));

	transposeKernel<<<blocks,threads>>>(dev_in, dev_out, xRes_in, yRes_in);

}