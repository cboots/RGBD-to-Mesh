#pragma once


#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "Calibration.h"
#include <glm/glm.hpp>
#include "Utils.h"
#include "math.h"


__host__ void computePlaneBoundingBoxes(PlaneStats planeStats, int* planeIdMap, int* numPlanesFound, 
										float4* aabbs, int* segments, int xRes, int yRes);