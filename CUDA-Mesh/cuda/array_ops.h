#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include "Calibration.h"
#include <glm/glm.hpp>
#include "Utils.h"
#include <math.h>


//Performs elementwise addition c = a + b;
__host__ void floatArrayAdditionCuda(float* dev_a, float* dev_b, float* dev_c, int size);

//Performs elementwise subtraction c = a - b;
__host__ void floatArraySubtractionCuda(float* dev_a, float* dev_b, float* dev_c, int size);

//Performs elementwise multiplication c = a * b;
__host__ void floatArrayMultiplyCuda(float* dev_a, float* dev_b, float* dev_c, int size);

//Performs elementwise division c = a / b;
__host__ void floatArrayDivisionCuda(float* dev_a, float* dev_b, float* dev_c, int size);