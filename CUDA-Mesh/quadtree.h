#pragma once


#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "Calibration.h"
#include <glm/glm.hpp>
#include "Utils.h"
#include "math.h"


#define AABB_COMPUTE_BLOCKWIDTH		32
#define AABB_COMPUTE_BLOCKHEIGHT	8


__host__ void computeAABBs(PlaneStats* planeStats, int* planeInvIdMap, glm::vec4* aabbsBlockResults,
						   int* planeCount, int maxPlanes,
						   Float3SOA positions, float* segmentProjectedSx, float* segmentProjectedSy, 
						   int* finalSegmentsBuffer, int xRes, int yRes);


__host__ void calculateProjectionData(rgbd::framework::Intrinsics intr, PlaneStats* planeStats,
									  int* planeCount, int maxTextureSize, int maxPlanes, int xRes, int yRes);


//Inputs are for a single texture projection, so first few inputs from arrays need to be preoffset to the correct plane
__host__ void projectTexture(int segmentId, PlaneStats* host_planeStats, PlaneStats* dev_planeStats, 
							 Float4SOA destTexture, int destTextureSize, 
							 RGBMapSOA rgbMap, int* dev_finalSegmentsBuffer, float* dev_finalDistanceToPlaneBuffer,
							 int imageXRes, int imageYRes);


__host__ void quadtreeDecimation(int actualWidth, int actualHeight, Float4SOA planarTexture, int* quadTreeAssemblyBuffer,
								 int textureBufferSize);

__host__ void quadtreeMeshGeneration(glm::vec4 aabbMeters, int actualWidth, int actualHeight, int* quadTreeAssemblyBuffer,
									 int* quadTreeScanResults, int textureBufferSize, int* blockResults, int blockResultsBufferSize,
									 int* indexBuffer, float4* vertexBuffer, int* compactCount, int* host_compactCount, int outputBufferSize,
									 int finalTextureWidth, int finalTextureHeight, Float4SOA planarTexture, float4* finalTexture);
