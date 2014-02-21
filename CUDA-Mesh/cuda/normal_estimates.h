#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include <glm/glm.hpp>


__host__ void simpleNormals(VMapSOA vmap, NMapSOA nmap, int numLevels, int xRes, int yRes);