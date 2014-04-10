#include "quadtree.h"

//Numthreads is assumed to be a power of two
__device__ void minmaxreduction(float* s_minSx, float* s_maxSx, float* s_minSy, float* s_maxSy, int indexInBlock, int nTotalThreads)
{
	int  thread2;
	float temp;

	while(nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (indexInBlock < halfPoint)
		{
			thread2 = indexInBlock + halfPoint;

			// Get the shared value stored by another thread
			temp = s_minSx[thread2];
			if (temp < s_minSx[indexInBlock]) 
				s_minSx[indexInBlock] = temp; 

			temp = s_minSy[thread2];
			if (temp < s_minSy[indexInBlock]) 
				s_minSy[indexInBlock] = temp; 

			temp = s_maxSx[thread2];
			if (temp > s_maxSx[indexInBlock]) 
				s_maxSx[indexInBlock] = temp; 

			temp = s_maxSy[thread2];
			if (temp > s_maxSy[indexInBlock]) 
				s_maxSy[indexInBlock] = temp; 
		}
		__syncthreads();

		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
}

__global__ void computeAABBsKernel(PlaneStats planeStats, int* planeInvIdMap, glm::vec3* tangents, glm::vec4* aabbs, 
								   int* planeCount, int maxPlanes,
								   Float3SOA positions, float* segmentProjectedSx, float* segmentProjectedSy, 
								   int* finalSegmentsBuffer, int xRes, int yRes)
{
	extern __shared__ int s_Mem[];
	int* s_InvMap = (int*) s_Mem;
	float* s_centroidX = (float*)(s_InvMap + maxPlanes);
	float* s_centroidY = s_centroidX + maxPlanes;
	float* s_centroidZ = s_centroidY + maxPlanes;
	glm::vec3* s_tangents = (glm::vec3*) (s_centroidZ + maxPlanes);
	glm::vec3* s_bitangents = s_tangents + maxPlanes;
	float* s_minSx = (float*)(s_bitangents + maxPlanes);
	float* s_minSy = (s_minSx + blockDim.x*blockDim.y);
	float* s_maxSx = (s_minSy + blockDim.x*blockDim.y);
	float* s_maxSy = (s_maxSx + blockDim.x*blockDim.y);

	int indexInBlock = threadIdx.x + threadIdx.y*blockDim.x;
	int imageX = threadIdx.x + blockDim.x*blockIdx.x;
	int imageY = threadIdx.y + blockDim.y*blockIdx.y;


	int numPlanes = planeCount[0];
	if(indexInBlock < maxPlanes)
	{
		s_InvMap[indexInBlock] = planeInvIdMap[indexInBlock];
		if(indexInBlock < numPlanes)
		{
			//s_aabb[indexInBlock] = glm::vec4(0.0f);
			s_tangents[indexInBlock] = tangents[indexInBlock];
			s_centroidX[indexInBlock] = planeStats.centroids.x[indexInBlock];
			s_centroidY[indexInBlock] = planeStats.centroids.y[indexInBlock];
			s_centroidZ[indexInBlock] = planeStats.centroids.z[indexInBlock];
			//bitangent = norm cross tangent
			glm::vec3 norm(planeStats.norms.x[indexInBlock],planeStats.norms.y[indexInBlock],planeStats.norms.z[indexInBlock]);
			s_bitangents[indexInBlock] = glm::normalize(glm::cross(norm, s_tangents[indexInBlock]));
		}
	}
	__syncthreads();

	//Remap segments
	int segment = finalSegmentsBuffer[imageX + imageY*xRes];
	float sx = 0;
	float sy = 0;
	if(segment >= 0)
	{
		//Remap and writeback
		segment = s_InvMap[segment];
		finalSegmentsBuffer[imageX + imageY*xRes] = segment;

		//Compute Sx and Sy
		glm::vec3 dp = glm::vec3(positions.x[imageX + imageY*xRes] - s_centroidX[segment], 
			positions.y[imageX + imageY*xRes] - s_centroidY[segment],
			positions.z[imageX + imageY*xRes] - s_centroidZ[segment]);

		sx = glm::dot(dp, s_bitangents[segment]);
		sy = glm::dot(dp, s_tangents[segment]);


	}
	//writeback
	segmentProjectedSx[imageX + imageY*xRes] = sx;
	segmentProjectedSy[imageX + imageY*xRes] = sy;

	__syncthreads();
	//Repurpose invmap sharedmem for segment flags

	if(indexInBlock < maxPlanes)
	{
		s_InvMap[indexInBlock] = 0;
	}

	__syncthreads();
	if(segment >= 0)//flag each segment that exists in this block
		s_InvMap[segment] = 1;

	for(int plane = 0; plane < numPlanes; ++plane)
	{
		if(s_InvMap[plane] > 0)
		{

			//Init minmax planes
			s_minSx[indexInBlock] = (segment == plane)?sx:0;
			s_maxSx[indexInBlock] = (segment == plane)?sx:0;
			s_minSy[indexInBlock] = (segment == plane)?sy:0;
			s_maxSy[indexInBlock] = (segment == plane)?sy:0;
			__syncthreads();
			minmaxreduction(s_minSx, s_maxSx, s_minSy, s_maxSy, indexInBlock, blockDim.x*blockDim.y);

		}
	}


}

__host__ void computeAABBs(PlaneStats planeStats, int* planeInvIdMap, glm::vec3* tangents, glm::vec4* aabbs, int* planeCount, int maxPlanes,
						   Float3SOA positions, float* segmentProjectedSx, float* segmentProjectedSy, 
						   int* finalSegmentsBuffer, int xRes, int yRes)
{
	int blockWidth = 32;
	int blockHeight = 8;

	assert(blockHeight*blockWidth >= maxPlanes);
	dim3 threads(blockWidth, blockHeight);
	dim3 blocks((int) ceil(xRes/float(blockWidth)), (int) ceil(yRes/float(blockHeight)));
	//plane map, tangent, bitangent, centroid and aabb of each plane loaded to shared memory.
	int sharedMem = maxPlanes*(sizeof(int) + sizeof(float)*3+sizeof(glm::vec3)*2) + blockWidth*blockHeight*4*sizeof(float);

	computeAABBsKernel<<<blocks,threads,sharedMem>>>(planeStats, planeInvIdMap, tangents, aabbs, planeCount, maxPlanes,
		positions, segmentProjectedSx, segmentProjectedSy, 
		finalSegmentsBuffer, xRes, yRes);

}