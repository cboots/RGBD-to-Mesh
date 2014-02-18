#pragma once
#include "cuda_runtime.h"
#include "RGBDFrame.h"
#include "device_structs.h"
#include <glm/glm.hpp>


// Draws depth image buffer to the texture.
// Texture width and height must match the resolution of the depth image.
// Returns false if width or height does not match, true otherwise
__host__ void drawDepthImageBufferToPBO(float4* dev_PBOpos, int texWidth, int texHeight);

// Draws color image buffer to the texture.
// Texture width and height must match the resolution of the color image.
// Returns false if width or height does not match, true otherwise
// dev_PBOpos must be a CUDA device pointer
__host__ void drawColorImageBufferToPBO(float4* dev_PBOpos, int texWidth, int texHeight);

// Renders various debug information about the 2D point cloud buffer to the texture.
// Texture width and height must match the resolution of the point cloud buffer.
// Returns false if width or height does not match, true otherwise
__host__ void drawPCBToPBO(float4* dptrPosition, float4* dptrColor, float4* dptrNormal, int texWidth, int texHeight);