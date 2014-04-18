#pragma once
#include <stdint.h>
#include <cuda.h>

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"

#define NUM_PYRAMID_LEVELS 3



struct RGBMapSOA{
	float* r;
	float* g;
	float* b;
};



struct Float3SOA{
	float* x;
	float* y;
	float* z;
};


struct Float4SOA{
	float* x;
	float* y;
	float* z;
	float* w;
};

struct Int3SOA{
	int* x;
	int* y;
	int* z;
};

struct Float1SOAPyramid{
	float* x[NUM_PYRAMID_LEVELS];
};

struct Float3SOAPyramid{
	float* x[NUM_PYRAMID_LEVELS];
	float* y[NUM_PYRAMID_LEVELS];
	float* z[NUM_PYRAMID_LEVELS];
};


struct PlaneStats{
	float* count;
	Float3SOA norms;
	Float3SOA eigs;
	Float3SOA centroids;
	//Scatter matrix decoupled
	float* Sxx;
	float* Syy;
	float* Szz;
	float* Sxy;
	float* Syz;
	float* Sxz;
};


struct ProjectionParameters{
	int destWidth;
	int destHeight;
	int textureResolution;//pixels/meter
	glm::mat3 projectionMatrix;
};

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