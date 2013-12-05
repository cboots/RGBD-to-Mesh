
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>

#include <stdio.h>
#include "device_structs.h"

/*
#define FOV_Y 43 # degrees
#define FOV_X 57

#define SCALE_Y tan((FOV_Y/2)*pi/180)
#define SCALE_X tan((FOV_X/2)*pi/180)
*/
#define SCALE_Y 0.393910475614942392
#define SCALE_X 0.542955699638436879

__global__ void makePointCloud(ColorPixel* colorPixels, DPixel* dPixels, int xRes, int yRes, PointCloud* pointCloud) {
    int i = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.y*blockDim.x) + (threadIdx.y*blockDim.x) + threadIdx.x;
    int r = i / xRes;
    int c = i % xRes;

    if (dPixels[i].depth > 0.0f) {
        float u = (c - (xRes-1)/2.0f + 1) / (xRes-1); // image plane u coordinate
        float v = ((yRes-1)/2.0f - r) / (yRes-1); // image plane v coordinate
        float Z = dPixels[i].depth/1000.0f; // depth in mm
        pointCloud[i].pos = glm::vec3(u*Z*SCALE_X, v*Z*SCALE_Y, Z); // convert uv to XYZ
        pointCloud[i].color = glm::vec3(colorPixels[i].r, colorPixels[i].g, colorPixels[i].b); // copy over texture
    }
}
