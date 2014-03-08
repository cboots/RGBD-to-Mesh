#include "plane_segmentation.h"

__global__ void normalHistogramKernel(float* normAzimuth, float* normPolar, int* histogram, int xRes, int yRes, int azimuthBins, int polarBins, 
									  float azimuthMultiplier, float polarMultiplier)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if( i  < xRes*yRes)
	{
		float polarAngle = normPolar[i];
		float azimuthAngle = normAzimuth[i];
		if(polarAngle == polarAngle && azimuthAngle == azimuthAngle)//Will be false if NaN
		{
			int pI = polarMultiplier*polarAngle;
			int aI = azimuthMultiplier*azimuthAngle;
			atomicAdd(&histogram[pI*azimuthBins + aI], 1);
		}
	}
}



__host__ void computeNormalHistogram(float* normAzimuth, float* normPolar, int* histogram, int xRes, int yRes, int azimuthBins, int polarBins)
{
	int blockLength = 256;
	
	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(xRes*yRes)/float(blockLength))));

	float azimuthMultiplier = azimuthBins/(2.0f*PI);
	float polarMultiplier = polarBins/(PI/2.0f);

	normalHistogramKernel<<<blocks,threads>>>(normAzimuth, normPolar, histogram, xRes, yRes, azimuthBins, polarBins, azimuthMultiplier, polarMultiplier);

}

__global__ void clearHistogramKernel(int* histogram, int length)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i < length)
	{
		histogram[i] = 0;
	}
}

__host__ void clearHistogram(int* histogram, int azimuthBins, int polarBins)
{
	int blockLength = 256;
	
	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(azimuthBins*polarBins)/float(blockLength))));

	clearHistogramKernel<<<blocks,threads>>>(histogram, azimuthBins*polarBins);

}