#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include <glm/glm.hpp>
#include "math.h"

__host__ void setSeperableKernelGaussian(int radius);

__host__ void setSeperableUniform();

__host__ void seperableFilter(float* x, float* y, float* z, float* x_out, float* y_out, float* z_out, 
								   int xRes, int yRes);
