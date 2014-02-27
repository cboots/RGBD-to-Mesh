#pragma once

#include <assert.h> 
#include "cuda_runtime.h"
#include <math.h>
#include "scan.h"
#include "transpose.h"

//Algorithm requires an intermediate buffer the same size as the input or output
__host__ void createIntegralImage(float* original, float* tempbuffer, float* integral, int width, int height);

/*
__device__ float AreaSum(float* integralImage, int imageWidth, int imageHeight, 
						int kernelLeft, int kernelRight, int kernelTop, int kernelBottom);
						*/