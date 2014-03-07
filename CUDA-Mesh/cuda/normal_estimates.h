#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include <glm/glm.hpp>
#include "math.h"

__host__ void simpleNormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, int numLevels, int xRes, int yRes);


__host__ void computeAverageGradientNormals(Float3SOAPyramid horizontalGradient, Float3SOAPyramid vertGradient, Float3SOAPyramid nmap, int xRes, int yRes);

__host__ void computePCANormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, float* curvature, int xRes, int yRes, float radiusMeters);

__host__ void convertNormalToSpherical(float* normX, float* normY, float* normZ, float* azimuthAngle, float* polarAngle, int arraySize);

__host__ void curvatureEstimate(Float3SOAPyramid nmap, float* curvature, int xRes, int yRes);