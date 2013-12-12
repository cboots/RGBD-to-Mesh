#pragma once
#include "cuda_runtime.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdint.h>
#include <stdio.h>
#include "RGBDFrame.h"
#include "device_structs.h"
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
/*
#define FOV_Y 43 # degrees
#define FOV_X 57

#define SCALE_Y tan((FOV_Y/2)*pi/180)
#define SCALE_X tan((FOV_X/2)*pi/180)
*/
#define SCALE_Y 0.393910475614942392
#define SCALE_X 0.542955699638436879
#define PI      3.141592653589793238
#define MIN_EIG_RATIO 1.0

#define RAD_WIN 1 // search window for nearest neighbors
#define RAD_NN 0.05 // nearest neighbor radius in world space (meters)
#define MIN_NN 3 // minimum number of nearest neighbors for valid normal

#define EPSILON 0.000000001

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

//Stream compacts only valid point cloud pixels into a VBO for efficient 3D rendering and the next pipeline stage.
//Returns number of elements in buffer when done
int compactPointCloudToVBO(PointCloud* vbo);

//Draws depth image buffer to the texture.
//Texture width and height must match the resolution of the depth image.
//Returns false if width or height does not match, true otherwise
bool drawDepthImageBufferToPBO(float4* pbo, int texWidth, int texHeight);

//Draws color image buffer to the texture.
//Texture width and height must match the resolution of the color image.
//Returns false if width or height does not match, true otherwise
bool drawColorImageBufferToPBO(float4* pbo, int texWidth, int texHeight);

//Renders various debug information about the 2D point cloud buffer to the texture.
//Texture width and height must match the resolution of the point cloud buffer.
//Returns false if width or height does not match, true otherwise
bool drawPCBToPBO(float4* dptrPosition, float4* dptrColor, float4* dptrNormal, int mXRes, int mYRes);

struct IsValidPoint
{
    template <typename T>
    __host__ __device__ __forceinline__
    bool operator() (const T a) const {
		return true;
        //return (glm::length(a.normal) > EPSILON);
    }
};