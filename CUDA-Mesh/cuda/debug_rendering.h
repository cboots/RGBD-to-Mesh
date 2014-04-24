#pragma once
#include "cuda_runtime.h"
#include "RGBDFrame.h"
#include "device_structs.h"
#include <glm/glm.hpp>
#include "math_constants.h"

using namespace rgbd::framework;

// Draws depth image buffer to the texture.
// Texture width and height must match the resolution of the depth image.
// Returns false if width or height does not match, true otherwise
__host__ void drawDepthImageBufferToPBO(float4* dev_PBOpos, DPixel* dev_colorImageBuffer, int texWidth, int texHeight);

// Draws color image buffer to the texture.
// Texture width and height must match the resolution of the color image.
// Returns false if width or height does not match, true otherwise
// dev_PBOpos must be a CUDA device pointer
__host__ void drawColorImageBufferToPBO(float4* dev_PBOpos, ColorPixel* dev_colorImageBuffer,  int texWidth, int texHeight);


__host__ void clearPBO(float4* pbo, int xRes, int yRes, float clearValue);

__host__ void drawVMaptoPBO(float4* pbo, Float3SOAPyramid vmap, int level, int xRes, int yRes);


__host__ void drawNMaptoPBO(float4* pbo, Float3SOAPyramid nmap, int level, int xRes, int yRes);


__host__ void drawRGBMaptoPBO(float4* pbo, Float3SOAPyramid rgbMap, int level, int xRes, int yRes);

__host__ void drawCurvaturetoPBO(float4* pbo, float* curvatureMap, int xRes, int yRes);

__host__ void drawNormalVoxelsToPBO(float4* pbo, int* voxels, int pboXRes, int pboYRes, int voxelAzimuthBins, int voxelPolarBins);

__host__ void drawDecoupledHistogramsToPBO(float4* pbo, Int3SOA histograms,  int length, int pboXRes, int pboYRes);

__host__ void drawNormalSegmentsToPBO(float4* pbo, int* normalSegments, float* projectedDistanceMap, 
									  int xRes, int yRes, int pboXRes, int pboYRes);

__host__ void drawScaledHistogramToPBO(float4* pbo, int* histogram, glm::vec3 color, int maxValue, int length, int pboXRes, int pboYRes);


__host__ void drawSegmentsDataToPBO(float4* pbo, int* normalSegments, float* projectedDistanceMap, float* projectedSx, float* projectedSy,
									  int xRes, int yRes, int pboXRes, int pboYRes);

__host__ void drawPlaneProjectedTexturetoPBO(float4* pbo, Float4SOA projectedTexture, int texWidth, int texHeight, 
											 int texBufferWidth, int pboXRes, int pboYRes);

__host__ void drawQuadtreetoPBO(float4* pbo, int* dev_quadTree, int texWidth, int texHeight, 
											 int texBufferWidth, int pboXRes, int pboYRes);
