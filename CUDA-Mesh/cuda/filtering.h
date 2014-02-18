#pragma once

#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include "Calibration.h"
#include <glm/glm.hpp>



__host__ void depthBufferToFloat(rgbd::framework::DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes);

__host__ void convertToPointCloud(float* dev_depthBuffer, rgbd::framework::ColorPixel* dev_colorPixels, 
								  PointCloud* dev_pointCloudBuffer,
								  int xRes, int yRes, rgbd::framework::Intrinsics intr, float maxDepth);