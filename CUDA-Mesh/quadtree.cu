#include "quadtree.h"


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
	glm::vec4* s_aabb = (glm::vec4*)(s_bitangents + maxPlanes);

	int indexInBlock = threadIdx.x + threadIdx.y*blockDim.x;
	int imageX = threadIdx.x + blockDim.x*blockIdx.x;
	int imageY = threadIdx.y + blockDim.y*blockIdx.y;


	int numPlanes = planeCount[0];
	if(indexInBlock < maxPlanes)
	{
		s_InvMap[indexInBlock] = planeInvIdMap[indexInBlock];
		if(indexInBlock < numPlanes)
		{
			s_aabb[indexInBlock] = glm::vec4(0.0f);
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
	if(segment >= 0)
	{
		//Remap and writeback
		segment = s_InvMap[segment];
		finalSegmentsBuffer[imageX + imageY*xRes] = segment;

		//Compute Sx and Sy
		glm::vec3 dp = glm::vec3(positions.x[imageX + imageY*xRes], positions.y[imageX + imageY*xRes], positions.z[imageX + imageY*xRes])
			- glm::vec3(s_centroidX[segment], s_centroidY[segment], s_centroidZ[segment]);

		float sx = glm::dot(dp, s_bitangents[segment]);
		float sy = glm::dot(dp, s_tangents[segment]);
		segmentProjectedSx[imageX + imageY*xRes] = sx;
		segmentProjectedSy[imageX + imageY*xRes] = sy;

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
	int sharedMem = maxPlanes*(sizeof(int) + sizeof(float)*3+sizeof(glm::vec3)*2 + sizeof(glm::vec4));

	computeAABBsKernel<<<blocks,threads,sharedMem>>>(planeStats, planeInvIdMap, tangents, aabbs, planeCount, maxPlanes,
		positions, segmentProjectedSx, segmentProjectedSy, 
		finalSegmentsBuffer, xRes, yRes);

}