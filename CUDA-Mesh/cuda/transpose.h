#pragma once

#include <assert.h> 
#include "cuda_runtime.h"
#include <math.h>


//Does a quick transpose of the input array and the output array.
__host__ void transpose(float* dev_in, float* dev_out, int xRes_in, int yRes_in);