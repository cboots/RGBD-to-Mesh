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
			//int xI = (x+1.0f)*0.5f*xBins;//x in range of -1 to 1. Map to 0 to 1.0 and multiply by number of bins
			//int yI = (y+1.0f)*0.5f*yBins;//x in range of -1 to 1. Map to 0 to 1.0 and multiply by number of bins
			//int xI = acos(x)*PI_INV_F*xBins;
			//int yI = acos(y)*PI_INV_F*yBins;
			float azimuth = acosf(x/sqrtf(1-y*y));
			int xI = azimuth*PI_INV_F*xBins;
			int yI = acos(y)*PI_INV_F*yBins;

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
		float angle = acosf(cosineValue[valueI]);

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
									 Int3SOA decoupledHistogram, int histSize, Int3SOA peakIndecies, int maxPeaks, int maxDistance)
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

	int histIndex = -1000000;

	float normalComponent = 0;
	int normI = threadIdx.x + blockDim.x * blockIdx.x;
	if(normI < imageWidth*imageHeight)
	{

		if(blockIdx.y == 0)	{
			normalComponent = rawNormals.x[normI];
		}else if(blockIdx.y == 1){
			normalComponent = rawNormals.y[normI];
		}else{
			normalComponent = rawNormals.z[normI];
		}

		float angle = acosf(normalComponent);

		if(angle == angle){
			histIndex = angle*PI_INV_F*histSize;
		}

		int minI = -1;
		int minDist = maxDistance;

		if(histIndex >= 0)
		{
			for(int i = 0; i < maxPeaks; i++)
			{
				if(s_peaks[i] < 0)
					break;

				int dist = s_peaks[i] - histIndex;
				dist = (dist > 0)?dist:-dist;//ABS value

				if(dist < minDist)
				{
					minI = i;
					minDist = dist;
				}
			}
		}

		if(blockIdx.y == 0)	{
			normalSegments.x[normI] = minI;
		}else if(blockIdx.y == 1){
			normalSegments.y[normI] = minI;
		}else{
			normalSegments.z[normI] = minI;
		}
	}

}

__host__ void segmentNormals(Float3SOA rawNormals, Int3SOA normalSegments, int imageWidth, int imageHeight, 
							 Int3SOA decoupledHistogram, int histSize, 
							 Int3SOA peakIndecies, int maxPeaks, int maxDistance)
{
	int blockLength = 512;

	assert(blockLength >= histSize);

	dim3 threads(blockLength);
	dim3 blocks((int) ceil(float(imageWidth*imageHeight)/float(blockLength)), 3);

	int sharedCount = sizeof(int)*(2 * maxPeaks);

	segmentNormalsKernel<<<blocks, threads, sharedCount>>>(rawNormals, normalSegments, imageWidth, imageHeight, 
		decoupledHistogram, histSize, 
		peakIndecies, maxPeaks, maxDistance);
}


#pragma endregion


#pragma region Histogram Peak Detection Two-D

__global__ void normalHistogramPrimaryPeakDetectionKernel(int* histogram, int xBins, int yBins, Float3SOA peaks, int maxPeaks, 
														  int exclusionRadius, int minPeakHeight)
{	
	extern __shared__ int s_temp[];
	int* s_hist = s_temp;
	int* s_max = s_hist + xBins*yBins;
	int* s_maxI = s_max + (xBins*yBins)/2;

	int index = threadIdx.x + threadIdx.y*xBins;
	//Load histogram
	s_hist[index] = histogram[index];
	__syncthreads();


	//Find local maxima
	bool localMax = false;

	if(s_hist[index] > minPeakHeight)
	{
		localMax = true;
		if(threadIdx.x > 0)
			if(s_hist[index - 1] > s_hist[index])
				localMax = false;


		if(threadIdx.x < xBins-1)
			if(s_hist[index + 1] > s_hist[index])
				localMax = false;

		if(threadIdx.y > 0)
			if(s_hist[index - xBins] > s_hist[index])
				localMax = false;

		if(threadIdx.y > yBins-1)
			if(s_hist[index + xBins] > s_hist[index])
				localMax = false;
	}

	float totalCount = 0.0f;
	float xPos = 0.0f;
	float yPos = 0.0f;
	if(localMax)
	{

		for(int x = -1; x <= 1; ++x)
		{
			int tx = threadIdx.x + x;
			for(int y = -1; y <= 1; ++y)
			{
				int ty = threadIdx.y + y;
				if(tx >= 0 && tx < xBins && ty >= 0 && ty < yBins)
				{
					int binCount = s_hist[tx + ty*xBins];
					totalCount += binCount;
					xPos += binCount*tx;
					yPos += binCount*ty;

				}
			}

		}
		xPos /= totalCount;
		yPos /= totalCount;

	}

	__syncthreads();

	if(!localMax)
	{
		s_hist[index] = 0;//clear all non-local max histograms

		//DEBUG
		//histogram[index] = 0;
	}
	__syncthreads();
	//Preprocessing complete

	//=========Peak detection Loop===========
	int histLength = xBins*yBins;
	for(int peakNum = 0; peakNum < maxPeaks; ++peakNum)
	{
		
#pragma region Maximum Finder
		//========Compute maximum=======
		//First step loads from main hist, so do outside loop
		int halfpoint = histLength >> 1;
		int thread2 = index + halfpoint;
		if(index < halfpoint)
		{
			int temp = s_hist[thread2];
			bool leftSmaller = (s_hist[index] < temp);
			s_max[index] = leftSmaller?temp:s_hist[index];
			s_maxI[index] = leftSmaller?thread2:index;
		}
		__syncthreads();
		while(halfpoint > 0)
		{
			halfpoint >>= 1;
			if(index < halfpoint)
			{
				thread2 = index + halfpoint;
				int temp = s_max[thread2];
				if (temp > s_max[index]) {
					s_max[index] = temp;
					s_maxI[index] = s_maxI[thread2];
				}
			}
			__syncthreads();
		}

		//========Compute maximum End=======
#pragma endregion



		//s_maxI[0] now holds the maximum index
		
		if(s_max[0] < minPeakHeight)
			break;

		if(s_maxI[0] == index)
		{
			peaks.x[peakNum] = xPos;
			peaks.y[peakNum] = yPos;
			peaks.z[peakNum] = totalCount;
			//DEBUG
			histogram[index] = -(peakNum+1);
		}

		//Distance to max
		int dx = (s_maxI[0] % xBins) - threadIdx.x;
		int dy = (s_maxI[0] / yBins) - threadIdx.y;

		if(dx*dx+dy*dy < exclusionRadius*exclusionRadius)
		{
			s_hist[index] = 0;
		}


		__syncthreads();
	}
}


__host__ void normalHistogramPrimaryPeakDetection(int* histogram, int xBins, int yBins, Float3SOA peaks, int maxPeaks, 
												  int exclusionRadius, int minPeakHeight)
{
	assert(xBins*yBins <= 1024);//For now enforce strict limit. Might be expandable in future, but most efficient like this
	assert(!(xBins*yBins  & (xBins*yBins  - 1))); //Assert is power of two



	dim3 threads(xBins, yBins);
	dim3 blocks(1);

	int sharedMem = xBins*yBins*2*sizeof(int);

	normalHistogramPrimaryPeakDetectionKernel<<<blocks,threads,sharedMem>>>(histogram, xBins, yBins, peaks, 
		maxPeaks, exclusionRadius, minPeakHeight);
}

#pragma endregion