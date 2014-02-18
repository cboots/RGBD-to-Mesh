#include "MeshTracker.h"

#pragma region Ctor/Dtor

MeshTracker::MeshTracker(int xResolution, int yResolution, Intrinsics intr)
{
	mXRes = xResolution;
	mYRes = yResolution;
	mIntr = intr;

	initCudaBuffers(mXRes, mYRes);

	resetTracker();
}


MeshTracker::~MeshTracker(void)
{
	cleanupCuda();
}
#pragma endregion

#pragma region Setup/Teardown functions
void MeshTracker::initCudaBuffers(int xRes, int yRes)
{
	cudaMalloc((void**) &dev_colorImageBuffer,				sizeof(ColorPixel)*xRes*yRes);
	cudaMalloc((void**) &dev_depthImageBuffer,				sizeof(DPixel)*xRes*yRes);
	cudaMalloc((void**) &dev_depthFilterIntermediateBuffer,	sizeof(float)*xRes*yRes);
	cudaMalloc((void**) &dev_pointCloudBuffer,				sizeof(PointCloud)*xRes*yRes);

}

void MeshTracker::cleanupCuda()
{
	cudaFree(dev_colorImageBuffer);
	cudaFree(dev_depthImageBuffer);
	cudaFree(dev_depthFilterIntermediateBuffer);
	cudaFree(dev_pointCloudBuffer);
}
#pragma endregion

#pragma region Pipeline control API
void MeshTracker::resetTracker()
{
	lastFrameTime = 0LL;
	currentFrameTime = 0LL;
	//TODO: Initalize and clear world tree


}


void MeshTracker::pushRGBDFrameToDevice(ColorPixelArray colorArray, DPixelArray depthArray, timestamp time)
{
	lastFrameTime = currentFrameTime;
	currentFrameTime = time;

	cudaMemcpy((void*)dev_depthImageBuffer, depthArray.get(), sizeof(DPixel)*mXRes*mYRes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)dev_colorImageBuffer, colorArray.get(), sizeof(ColorPixel)*mXRes*mYRes, cudaMemcpyHostToDevice);

}


void MeshTracker::depthToFloatNoFilter()
{
	depthBufferToFloat(dev_depthImageBuffer, dev_depthFilterIntermediateBuffer, mXRes, mYRes);
}


void MeshTracker::assemblePointCloud(float maxDepth)
{
	convertToPointCloud(dev_depthFilterIntermediateBuffer, dev_colorImageBuffer, dev_pointCloudBuffer,
								  mXRes, mYRes, mIntr, maxDepth);
}

#pragma endregion