#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include "Calibration.h"
#include <glm/glm.hpp>
#include "Utils.h"
#include "math.h"

__host__ void computeNormalHistogram(float* normX, float* normY, int* histogram, int xRes, int yRes, int xBins, int yBins);
__host__ void clearHistogram(int* histogram, int xBins, int yBins);


__host__ void ACosHistogram(float* cosineValue, int* histogram, int valueCount, int numBins);