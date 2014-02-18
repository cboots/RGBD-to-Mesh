#pragma once
#include "RGBDFrame.h"
#include "device_structs.h"
#include "cuda_runtime.h"
#include "filtering.h"

using namespace rgbd::framework;

enum FilterMode
{
	BILATERAL_FILTER,
	GAUSSIAN_FILTER,
	NO_FILTER
};

class MeshTracker
{
private:
	//META DATA
#pragma region State Variables
	timestamp lastFrameTime;
	timestamp currentFrameTime;
#pragma endregion

#pragma region Configuration Variables
	int mXRes;
	int mYRes;
	Intrinsics mIntr;
#pragma region

#pragma region Pipeline Buffer Device Pointers
	//PIPELINE BUFFERS
	ColorPixel* dev_colorImageBuffer;
	DPixel* dev_depthImageBuffer;
	float* dev_depthFilterIntermediateBuffer;

	PointCloud* dev_pointCloudBuffer;
#pragma region
	
#pragma region Private Methods
	void initCudaBuffers(int xRes, int yResolution);
	void cleanupCuda();
#pragma endregion

public:

#pragma region Ctor/Dtor
	MeshTracker(int xResolution, int yResolution, Intrinsics intr);
	~MeshTracker(void);
#pragma endregion

#pragma region Public API
	//Setup and clear tracked world model
	void resetTracker();

	void pushRGBDFrameToDevice(ColorPixelArray colorArray, DPixelArray depthArray, timestamp time);

	void depthToFloatNoFilter();

	void assemblePointCloud(float maxDepth);
#pragma endregion

#pragma region Buffer getters
	inline PointCloud* getPCBDevicePtr() { return dev_pointCloudBuffer;}
	inline ColorPixel* getColorImageDevicePtr() { return dev_colorImageBuffer;}
	inline DPixel* getDepthImageDevicePtr() { return dev_depthImageBuffer;}
#pragma endregion
};

