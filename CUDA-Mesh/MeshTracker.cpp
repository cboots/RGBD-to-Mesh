#include "MeshTracker.h"


MeshTracker::MeshTracker(int xResolution, int yResolution)
{
	mXRes = xResolution;
	mYRes = yResolution;

	initCuda(mXRes, mYRes);

	resetTracker();
}


MeshTracker::~MeshTracker(void)
{
	cleanupCuda();
}


void MeshTracker::resetTracker()
{
	lastFrameTime = 0LL;
	currentFrameTime = 0LL;
	//TODO: Initalize and clear world tree


}


void MeshTracker::pushRGBDFrameToDevice(ColorPixelArray colorArray, DPixelArray depthArray, timestamp time)
{
	pushColorArrayToBuffer(colorArray.get(), mXRes, mYRes);
	pushDepthArrayToBuffer(depthArray.get(), mXRes, mYRes);

	lastFrameTime = currentFrameTime;
	currentFrameTime = time;
}