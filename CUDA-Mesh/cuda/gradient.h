#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "math.h"


__host__ void horizontalGradient(float* image_in, float* gradient_out, int width, int height);
__host__ void verticalGradient(float* image_in, float* gradient_out, int width, int height);