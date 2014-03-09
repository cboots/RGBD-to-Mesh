#include "plane_segmentation.h"


#pragma region Histogram Two-D

__global__ void normalHistogramKernel(float* normX, float* normY, int* histogram, int xRes, int yRes, int xBins, int yBins)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if( i  < xRes*yRes)
	{
		float x = normX[i];
		float y = normY[i];
		if(x == x && y == y)//Will be false if NaN
		{
			int xI = (x+1.0f)*0.5f*xBins;//x in range of -1 to 1. Map to 0 to 1.0 and multiply by number of bins
			int yI = (y+1.0f)*0.5f*yBins;//x in range of -1 to 1. Map to 0 to 1.0 and multiply by number of bins
			atomicAdd(&histogram[yI*xBins + xI], 1);
		}
	}
}



__host__ void computeNormalHistogram(float* normX, float* normY, int* histogram, int xRes, int yRes, int xBins, int yBins)
{
	int blockLength = 256;
	
	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(xRes*yRes)/float(blockLength))));


	normalHistogramKernel<<<blocks,threads>>>(normX, normY, histogram, xRes, yRes, xBins, yBins);

}

__global__ void clearHistogramKernel(int* histogram, int length)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i < length)
	{
		histogram[i] = 0;
	}
}

__host__ void clearHistogram(int* histogram, int xBins, int yBins)
{
	int blockLength = 256;
	
	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(xBins*yBins)/float(blockLength))));

	clearHistogramKernel<<<blocks,threads>>>(histogram, xBins*yBins);

}

#pragma endregion


#pragma region ACos Histogram One-D

//TODO: Shared memory
__global__ void ACosHistogramKernel(float* cosineValue, int* histogram, int valueCount, int numBins)
{
	int valueI = threadIdx.x + blockDim.x * blockIdx.x;

	if(valueI < valueCount)
	{
		float angle = acos(cosineValue[valueI]);

		int histIndex = angle*PI_INV_F*numBins;
		if(histIndex >= 0 && histIndex < numBins)//Sanity check
			atomicAdd(&histogram[histIndex], 1);

	}
}

__host__ void ACosHistogram(float* cosineValue, int* histogram, int valueCount, int numBins)
{
	int blockLength = 256;

	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(valueCount)/float(blockLength))));

	ACosHistogramKernel<<<blocks,threads>>>(cosineValue, histogram, valueCount, numBins);
}

#pragma endregion