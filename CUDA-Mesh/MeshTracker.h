#pragma once
#include "RGBDFrame.h"
#include "device_structs.h"
#include "cuda_runtime.h"

using namespace rgbd::framework;
class MeshTracker
{
private:
	//META DATA
	timestamp lastFrameTime;
	timestamp currentFrameTime;
	int mXRes;
	int mYRes;

	//PIPELINE BUFFERS
	ColorPixel* dev_colorImageBuffer;
	DPixel* dev_depthImageBuffer;
	float* dev_depthFilterIntermediateBuffer;

	PointCloud* dev_pointCloudBuffer;

public:
	MeshTracker(int xResolution, int yResolution);
	~MeshTracker(void);

	//Setup and clear tracked world model
	void resetTracker();

	void pushRGBDFrameToDevice(ColorPixelArray colorArray, DPixelArray depthArray, timestamp time);

	void initCudaBuffers(int xRes, int yResolution);
	void cleanupCuda();

	inline PointCloud* getPCBDevicePtr() { return dev_pointCloudBuffer;}
	inline ColorPixel* getColorImageDevicePtr() { return dev_colorImageBuffer;}
	inline DPixel* getDepthImageDevicePtr() { return dev_depthImageBuffer;}
};

