#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include <glm/glm.hpp>

__host__ void simpleNormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, Float1SOAPyramid curvaturemap, int numLevels, int xRes, int yRes);


__host__ void computeAverageGradientNormals(Float3SOAPyramid horizontalGradient, Float3SOAPyramid vertGradient, Float3SOAPyramid nmap, 
											Float1SOAPyramid curvature, int xRes, int yRes);

__host__ void computePCANormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, Float1SOAPyramid curvaturemap, 
								int xRes, int yRes, float radiusMeters, int radiusPixels, int minNeighbors);