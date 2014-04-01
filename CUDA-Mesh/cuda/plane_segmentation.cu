#include "plane_segmentation.h"


__device__ glm::vec3 normalFrom3x3Covar(glm::mat3 A, float& curvature) {
	// Given a (real, symmetric) 3x3 covariance matrix A, returns the eigenvector corresponding to the min eigenvalue
	// (see: http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices)
	glm::vec3 eigs;
	glm::vec3 normal = glm::vec3(0.0f);

	float p1 = A[0][1]*A[0][1] + A[0][2]*A[0][2] + A[1][2]*A[1][2];
	if (abs(p1) < 0.00001f) { // A is diagonal
		eigs = glm::vec3(A[0][0], A[1][1], A[2][2]);

		float tmp;
		int i, eig_i;
		// sorting: swap first pair if necessary, then second pair, then first pair again
		for (i=0; i<3; i++) {
			eig_i = i%2;
			tmp = eigs[eig_i];
			eigs[eig_i] = glm::max(tmp, eigs[eig_i+1]);
			eigs[eig_i+1] = glm::min(tmp, eigs[eig_i+1]);
		}
	} else {
		float q = (A[0][0] + A[1][1] + A[2][2])/3.0f; // mean(trace(A))
		float p2 = (A[0][0]-q)*(A[0][0]-q) + (A[1][1]-q)*(A[1][1]-q) + (A[2][2]-q)*(A[2][2]-q)+ 2*p1;
		float p = sqrt(p2/6);
		glm::mat3 B = (1/p) * (A-q*glm::mat3(1.0f));
		float r = glm::determinant(B)/2;
		// theoretically -1 <= r <= 1, but clamp in case of numeric error
		float phi;
		if (r <= -1) {
			phi = PI_F / 3;
		} else if (r >= 1) { 
			phi = 0;
		} else {
			phi = glm::acos(r)/3;
		}
		eigs[0] = q + 2*p*glm::cos(phi);
		eigs[2] = q + 2*p*glm::cos(phi + 2*PI_F/3);
		eigs[1] = 3*q - eigs[0] - eigs[2];

	}



	//N = (A-eye(3)*eig1)*(A(:,1)-[1;0;0]*eig2);
	glm::mat3 Aeig1 = A;
	Aeig1[0][0] -= eigs[0];
	Aeig1[1][1] -= eigs[0];
	Aeig1[2][2] -= eigs[0];
	normal = Aeig1*(A[0] - glm::vec3(eigs[1],0.0f,0.0f));

	// check if point cloud region is "flat" enough
	curvature = eigs[2]/(eigs[0]+eigs[1]+eigs[2]);


	float length = glm::length(normal);
	normal /= length;
	return normal;
}


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
			float azimuth = acosf(x/sqrtf(1.0f-y*y));
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
		{
			//Fill remaining slots with -1
			if(index >= peakNum && index < maxPeaks)
			{
				peaks.x[index] = -1;
				peaks.y[index] = -1;
				peaks.z[index] = -1;
			}
			break;
		}

		if(s_maxI[0] == index)
		{
			peaks.x[peakNum] = xPos;
			peaks.y[peakNum] = yPos;
			peaks.z[peakNum] = s_hist[index];
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

#pragma region Segmentation Two-D

__global__ void segmentNormals2DKernel(Float3SOA rawNormals, Float3SOA rawPositions, 
									   int* normalSegments, float* projectedDistance,
									   int imageWidth, int imageHeight, 
									   int* histogram, int xBins, int yBins, 
									   Float3SOA peaks, int maxPeaks, float maxAngleRange)
{
	extern __shared__ float s_mem[];
	float* s_peaksX = s_mem;
	float* s_peaksY = s_peaksX + maxPeaks;
	float* s_peaksZ = s_peaksY + maxPeaks;

	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if(threadIdx.x < maxPeaks)
	{
		float xi = peaks.x[threadIdx.x];
		float yi = peaks.y[threadIdx.x];
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;

		if(xi >= 0.0f && yi >= 0.0f){

			y = cosf(PI_F*yi/float(yBins));
			x = cosf(PI_F*xi/float(xBins)) * sqrtf(1.0f-y*y);
			z = sqrtf(1.0f-x*x-y*y);
		}

		s_peaksX[threadIdx.x] = x;
		s_peaksY[threadIdx.x] = y;
		s_peaksZ[threadIdx.x] = z;
	}

	__syncthreads();


	if(index < imageWidth*imageHeight)
	{

		glm::vec3 normal = glm::vec3(rawNormals.x[index], rawNormals.y[index], rawNormals.z[index]);
		int bestPeak = -1;
		if(normal.x == normal.x && normal.y == normal.y && normal.z == normal.z)
		{
			//normal is valid
			for(int peakNum = 0; peakNum < maxPeaks; ++peakNum)
			{
				float dotprod = normal.x*s_peaksX[peakNum] + normal.y*s_peaksY[peakNum] + normal.z*s_peaksZ[peakNum];
				float angle = acosf(dotprod);

				if(angle < maxAngleRange)
				{
					bestPeak = peakNum;
					break;
				}
			}
		}

		float projectedD = CUDART_NAN_F;//Initialize to NAN
		if(bestPeak >= 0)
		{
			//Peak found, compute projection
			projectedD = s_peaksX[bestPeak]*rawPositions.x[index] 
			+ s_peaksY[bestPeak]*rawPositions.y[index] 
			+ s_peaksZ[bestPeak]*rawPositions.z[index];

		}


		//Writeback
		normalSegments[index] = bestPeak;
		projectedDistance[index] = projectedD;
	}

}

__host__ void segmentNormals2D(Float3SOA rawNormals, Float3SOA rawPositions, 
							   int* normalSegments, float* projectedDistance,int imageWidth, int imageHeight,
							   int* normalHistogram, int xBins, int yBins, 
							   Float3SOA peaks, int maxPeaks, float maxAngleRange)
{
	int blockLength = 512;
	assert(blockLength > maxPeaks);

	dim3 blocks((int) ceil(float(imageWidth*imageHeight)/float(blockLength)));
	dim3 threads(blockLength);

	int sharedCount = sizeof(float)*(3 * maxPeaks);

	segmentNormals2DKernel<<<blocks, threads, sharedCount>>>(rawNormals, rawPositions, normalSegments, projectedDistance, 
		imageWidth, imageHeight, normalHistogram, xBins, yBins, peaks, maxPeaks, maxAngleRange);
}


#pragma endregion

#pragma region Distance Histograms

__global__ void distanceHistogramKernel(int* dev_normalSegments, float* dev_planeProjectedDistanceMap, int xRes, int yRes,
										 int* dev_distanceHistograms, int numMaxNormalSegments, 
										 int histcount, float histMinDist, float histMaxDist)
{
	extern __shared__ int s_temp[];
	int* s_hist = s_temp;

	int index = threadIdx.x + blockIdx.x*blockDim.x;

	int segment = dev_normalSegments[index];
	float dist = dev_planeProjectedDistanceMap[index];
	int histI = -1;
	if(segment >= 0)
	{
		if(dist < histMaxDist && dist >= histMinDist) 
			histI = (dist - histMinDist)*histcount/(histMaxDist-histMinDist);
	}

	//Each thread has locally stored values.
	for(int peak = 0; peak < numMaxNormalSegments; ++peak)
	{
		//reset histogram
		s_temp[threadIdx.x] = 0;
		__syncthreads();

		if(segment == peak && histI >= 0)
		{
			atomicAdd(&s_hist[histI], 1);
		}
	
		__syncthreads();

		atomicAdd(&(dev_distanceHistograms[peak*histcount + threadIdx.x]), s_hist[threadIdx.x]);
	}
}

__host__ void generateDistanceHistograms(int* dev_normalSegments, float* dev_planeProjectedDistanceMap, int xRes, int yRes,
										 int** dev_distanceHistograms, int numMaxNormalSegments, 
										 int histcount, float histMinDist, float histMaxDist)
{
	int blockLength = histcount;

	assert(xRes*yRes % blockLength == 0);//Assert even division, otherwise kernel will crash.

	dim3 threads(blockLength);
	dim3 blocks((int)(ceil(float(xRes*yRes)/float(blockLength))));

	int sharedSize = histcount * sizeof(int);

	distanceHistogramKernel<<<blocks,threads,sharedSize>>>(dev_normalSegments, dev_planeProjectedDistanceMap, xRes, yRes, 
		dev_distanceHistograms[0], numMaxNormalSegments, histcount, histMinDist, histMaxDist);
}

#pragma endregion