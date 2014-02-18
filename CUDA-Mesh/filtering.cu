#include "filtering.h"

__global__ void depthBufferToFloatKernel(DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes) {
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * xRes) + c;

	if(r < yRes && c < xRes) {
		dev_depthFloat[i] = dev_depthBuffer[i].depth*0.001f; // depth converted from mm to meters
	}
}

__host__ void depthBufferToFloat(DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes)
{

	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	depthBufferToFloatKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_depthBuffer, dev_depthFloat, xRes, yRes);

}

