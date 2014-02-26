#pragma once

#include <assert.h> 
#include "cuda_runtime.h"
#include <math.h>


//Performs exclusive scan on rows of a matrix of (width < 1024)
//Can be an in place scan by setting dev_in == dev_out
__host__ void exclusiveScanRows(float* dev_in, float* dev_out, int width, int height);