#pragma once
#include <stdint.h>
#include <cuda.h>
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"

typedef struct
{
	// Point location
	glm::vec3 pos;
	// RGB point color
	glm::vec3 color;
	// Normalized point normal vector
	glm::vec3 normal;
} PointCloud;

typedef struct
{
    glm::vec3 eigenVals;
    glm::mat3 eigenVecs;
} EigenResult;
