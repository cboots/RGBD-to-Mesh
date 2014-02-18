#pragma once

#include "cuda_runtime.h"
#include "RGBDFrame.h"
#include "device_structs.h"
#include <glm/glm.hpp>

using namespace rgbd::framework;


__host__ void depthBufferToFloat(DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes);