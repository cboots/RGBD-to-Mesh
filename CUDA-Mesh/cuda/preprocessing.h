#pragma once

#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include "Calibration.h"
#include <glm/glm.hpp>



__host__ void depthBufferToFloat(rgbd::framework::DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes);

__host__ void rgbAOSToSOA(rgbd::framework::ColorPixel* dev_colorPixels, 
						  RGBMapSOA rgbSOA, int xRes, int yRes);