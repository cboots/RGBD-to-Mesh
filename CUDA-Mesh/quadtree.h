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


__host__ void computeAABBs(PlaneStats planeStats, int* planeInvIdMap, glm::vec3* tangents, glm::vec4* aabbs, glm::vec4* aabbsBlockResults,
						   int* planeCount, int maxPlanes,
						   Float3SOA positions, float* segmentProjectedSx, float* segmentProjectedSy, 
						   int* finalSegmentsBuffer, int xRes, int yRes);


__host__ void calculateProjectionData(rgbd::framework::Intrinsics intr, PlaneStats planeStats, glm::vec3* tangents, glm::vec4* aabbs, 
									  ProjectionParameters* projParams, int* planeCount, int maxPlanes, int xRes, int yRes);