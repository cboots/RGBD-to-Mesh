#pragma once
#include "RGBDFrame.h"
#include "kernel.h"

using namespace rgbd::framework;
class MeshTracker
{
private:
	timestamp lastFrameTime;
	timestamp currentFrameTime;
	int mXRes;
	int mYRes;
public:
	MeshTracker(int xResolution, int yResolution);
	~MeshTracker(void);

	//Setup and clear tracked world model
	void resetTracker();

	void pushRGBDFrameToDevice(ColorPixelArray colorArray, DPixelArray depthArray, timestamp time);
};

