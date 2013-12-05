#pragma once
#include "RGBDFrame.h"
#include "device_structs.h"

// THIS IS FOR ALL THE DEVICE KERNEL WRAPPER FUNCTIONS

void initCuda(int width, int height);
void cleanupCuda();


//Copies a depth image to the GPU buffer. 
//Returns false if width and height do not match buffer size set by initCuda(), true if success
bool pushDepthArrayToBuffer(DPixel* hDepthArray, int width, int height);


//Copies a color image to the GPU buffer. 
//Returns false if width and height do not match buffer size set by initCuda(), true if success
bool pushColorArrayToBuffer(ColorPixel* hColorArray, int width, int height);

//Converts the color and depth images currently in GPU buffers into point cloud buffer
bool convertToPointCloud();