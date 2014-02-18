#pragma once
#include <stdint.h>
#include <cuda.h>

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"


typedef struct PointCloud
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
	unsigned int v0;
	unsigned int v1;
	unsigned int v2;
} triangleIndecies;

typedef struct {
	unsigned int vertex_array;
	unsigned int vbo_indices;
	unsigned int num_indices;
	//Don't need these to get it working, but needed for deallocation
	unsigned int vbo_data;
} device_mesh2_t;


typedef struct {
	glm::vec3 pt;
	glm::vec2 texcoord;
} vertex2_t;