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
	extern __shared__ int s_hist[];
	s_hist[threadIdx.x] = 0;
	__syncthreads();

	int valueI = threadIdx.x + blockDim.x * blockIdx.x;

	if(valueI < valueCount)
	{
		float angle = acos(cosineValue[valueI]);

		if(angle == angle){
			int histIndex = angle*PI_INV_F*numBins;
			if(histIndex >= 0 && histIndex < numBins)//Sanity check
				atomicAdd(&s_hist[histIndex], 1);
		}
	}

	__syncthreads();

	atomicAdd(&histogram[threadIdx.x], s_hist[threadIdx.x]);
}

__host__ void ACosHistogram(float* cosineValue, int* histogram, int valueCount, int numBins)
{
	int blockLength = numBins;

	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(valueCount)/float(blockLength))));

	ACosHistogramKernel<<<blocks,threads, numBins*sizeof(int)>>>(cosineValue, histogram, valueCount, numBins);
}

#pragma endregion


#pragma region Simple Histogram Peak Detection

__global__ void gaussianSubtractionPeakDetectionKernel(int* histx, int* histy, int* histz, int* peaksX, int* peaksY, int* peaksZ, 
													   int histLength, int maxPeaks, int minPeakCount, glm::vec3 sigma2inv)
{
	//Setup shared buffers
	extern __shared__ int s_temp[];
	int* s_hist = s_temp;
	int* s_max = s_hist + histLength;
	int* s_maxI = s_max + histLength/2;
	int* s_peaks = s_maxI + histLength/2;

	//Load histogram from different location for each block
	float sig = sigma2inv[blockIdx.x];
	if(blockIdx.x == 0)
		s_hist[threadIdx.x] = histx[threadIdx.x];
	else if(blockIdx.x == 1)
		s_hist[threadIdx.x] = histy[threadIdx.x];
	else //if(blockIdx.x == 2)
		s_hist[threadIdx.x] = histz[threadIdx.x];

	//clear peaks
	if(threadIdx.x < maxPeaks)
		s_peaks[threadIdx.x] = -1;

	__syncthreads();
	//====Load/Init Complete=====
	//====Begin Peak Loop =======

	//For up to the maximum number of peaks
	for(int peaki = 0; peaki < maxPeaks; ++peaki)
	{

#pragma region Maximum Finder
		//========Compute maximum=======
		//First step loads from main hist, so do outside loop
		int halfpoint = histLength >> 1;
		int thread2 = threadIdx.x + halfpoint;

		if(threadIdx.x < halfpoint)
		{
			int temp = s_hist[thread2];
			bool leftSmaller = (s_hist[threadIdx.x] < temp);
			s_max[threadIdx.x] = leftSmaller?temp:s_hist[threadIdx.x];
			s_maxI[threadIdx.x] = leftSmaller?thread2:threadIdx.x;
		}
		__syncthreads();
		while(halfpoint > 0)
		{
			halfpoint >>= 1;
			if(threadIdx.x < halfpoint)
			{
				thread2 = threadIdx.x + halfpoint;
				int temp = s_max[thread2];
				if (temp > s_max[threadIdx.x]) {
					s_max[threadIdx.x] = temp;
					s_maxI[threadIdx.x] = s_maxI[thread2];
				}
			}
			__syncthreads();
		}

		//========Compute maximum End=======
#pragma endregion

		if(threadIdx.x == 0)
		{
			s_peaks[peaki] = s_maxI[0];
		}

		if(s_max[0] < minPeakCount)
			break;//done. No more peaks to find

		//=====Subtract gaussian model=====
		int diff = (threadIdx.x-s_maxI[peaki]);
		s_hist[threadIdx.x] -= s_max[0] * expf(-diff*diff*sig);

		__syncthreads();
	}

	//Writeback
	if(threadIdx.x < maxPeaks)
	{
		if(blockIdx.x == 0)
			peaksX[threadIdx.x] = s_peaks[threadIdx.x];
		else if(blockIdx.x == 1)
			peaksY[threadIdx.x] = s_peaks[threadIdx.x];
		else //if(blockIdx.x == 2)
			peaksZ[threadIdx.x] = s_peaks[threadIdx.x];
	}
}

__host__ void gaussianSubtractionPeakDetection(Int3SOA decoupledHist, Int3SOA peakIndex, int histSize, int maxPeaks, int minPeakCount, glm::vec3 sigmas)
{
	assert(histSize > 32);
	assert(!(histSize & (histSize - 1))); //Assert is power of two
	assert(histSize % 32 == 0);//Assert is multiple of 32

	int sharedSize = (histSize*2 + maxPeaks)*sizeof(int);
	dim3 threads(histSize);
	dim3 blocks(3);

	gaussianSubtractionPeakDetectionKernel<<<blocks,threads,sharedSize>>>(decoupledHist.x, decoupledHist.y, decoupledHist.z, 
		peakIndex.x, peakIndex.y, peakIndex.z, histSize, maxPeaks, minPeakCount, 1.0f/(2.0f*sigmas*sigmas));

}


#pragma endregion

#pragma region Seperable histogram segmentation


__global__ void segmentNormalsKernel(Float3SOA rawNormals, Int3SOA normalSegments, int imageWidth, int imageHeight, 
									 Int3SOA decoupledHistogram, int histSize, Int3SOA peakIndecies, int maxPeaks)
{
	extern __shared__ int s_temp[];
	int* s_peaks = s_temp;
	int* s_peaksMax = s_temp + maxPeaks;


	if(threadIdx.x < maxPeaks)
	{
		if(blockIdx.y == 0)	{
			s_peaks[threadIdx.x] = peakIndecies.x[threadIdx.x];
			s_peaksMax[threadIdx.x] = (s_peaks[threadIdx.x] < 0)? 0 : decoupledHistogram.x[s_peaks[threadIdx.x]];
		}else if(blockIdx.y == 1){
			s_peaks[threadIdx.x] = peakIndecies.y[threadIdx.x];
			s_peaksMax[threadIdx.x] = (s_peaks[threadIdx.x] < 0)? 0 : decoupledHistogram.y[s_peaks[threadIdx.x]];
		}else{
			s_peaks[threadIdx.x] = peakIndecies.z[threadIdx.x];
			s_peaksMax[threadIdx.x] = (s_peaks[threadIdx.x] < 0)? 0 : decoupledHistogram.z[s_peaks[threadIdx.x]];
		}

	}

	__syncthreads();


}

__host__ void segmentNormals(Float3SOA rawNormals, Int3SOA normalSegments, int imageWidth, int imageHeight, 
							 Int3SOA decoupledHistogram, int histSize, 
							 Int3SOA peakIndecies, int maxPeaks)
{
	int blockLength = 512;

	dim3 threads(blockLength);
	dim3 blocks((int) ceil(float(imageWidth*imageHeight)/float(blockLength)), 3);

	int sharedCount = sizeof(int)*(2 * maxPeaks);

	segmentNormalsKernel<<<blocks, threads, sharedCount>>>(rawNormals, normalSegments, imageWidth, imageHeight, 
		decoupledHistogram, histSize, 
		peakIndecies, maxPeaks);
}


#pragma endregion