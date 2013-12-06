#pragma once
#include <GL/glew.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include "RGBDFrame.h"
#include "device_structs.h"
#include <glm/glm.hpp>


// THIS IS FOR ALL THE DEVICE KERNEL WRAPPER FUNCTIONS

//Check for error
void checkCUDAError(const char *msg) ;

//Intialize pipeline buffers
void initCuda(int width, int height);

//Free all allocated buffers and close out environment
void cleanupCuda();


//Copies a depth image to the GPU buffer. 
//Returns false if width and height do not match buffer size set by initCuda(), true if success
bool pushDepthArrayToBuffer(DPixel* hDepthArray, int width, int height);


//Copies a color image to the GPU buffer. 
//Returns false if width and height do not match buffer size set by initCuda(), true if success
bool pushColorArrayToBuffer(ColorPixel* hColorArray, int width, int height);

//Converts the color and depth images currently in GPU buffers into point cloud buffer
void convertToPointCloud();

//Computes normals for point cloud in buffer and writes back to the point cloud buffer.
void computePointCloudNormals();


//Draws depth image buffer to the texture.
//Texture width and height must match the resolution of the depth image.
//Returns false if width or height does not match, true otherwise
bool drawDepthImageBufferToTexture(GLuint texture, int texWidth, int texHeight);

//Draws color image buffer to the texture.
//Texture width and height must match the resolution of the color image.
//Returns false if width or height does not match, true otherwise
bool drawColorImageBufferToTexture(GLuint texture, int texWidth, int texHeight);

//Renders the point cloud as stored in the VBO to the texture
void drawPointCloudVBOToTexture(GLuint texture, int texWidth, int texHeight /*TODO: More vizualization parameters here*/);

//Renders various debug information about the 2D point cloud buffer to the texture.
//Texture width and height must match the resolution of the point cloud buffer.
//Returns false if width or height does not match, true otherwise
bool drawPointCloudDebugToTexture(GLuint texture, int texWidth, int texHeight /*TODO: More vizualization parameters here*/);