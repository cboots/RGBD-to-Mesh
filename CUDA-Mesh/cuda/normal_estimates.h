#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include <glm/glm.hpp>

__host__ void simpleNormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, Float1SOAPyramid curvaturemap, int numLevels, int xRes, int yRes);


__host__ void computeAverageGradientNormals(Float3SOAPyramid horizontalGradientII, Float3SOAPyramid vertGradientII, Float3SOAPyramid nmap, Float1SOAPyramid curvature, int level, int xRes, int yRes, int smoothRadius);