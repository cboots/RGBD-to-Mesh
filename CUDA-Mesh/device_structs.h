#pragma once
#include <stdint.h>
#include "glm/glm.hpp"

typedef struct
{
	/* Red value of this pixel. */
	uint8_t r;
	/* Green value of this pixel. */
	uint8_t g;
	/* Blue value of this pixel. */
	uint8_t b;
} ColorPixel;

typedef struct
{
	/* Depth value of this pixel */
	uint16_t depth;
} DPixel;

typedef struct
{
	/* Point location */
	glm::vec3 pos;
	/* RGB point color */
	glm::vec3 color;
	/* Normalized point normal vector */
	glm::vec3 normal;
} PointCloud;