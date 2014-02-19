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
	int pixCount = xRes*yRes;
	cudaMalloc((void**) &dev_colorImageBuffer,				sizeof(ColorPixel)*pixCount);
	cudaMalloc((void**) &dev_depthImageBuffer,				sizeof(DPixel)*pixCount);


	//Setup SOA buffers, ensuring contiguous memory for pyramids 
	//RGB SOA
	cudaMalloc((void**) &dev_rgbSOA.r, sizeof(float)*3*pixCount);//Allocate enough for r,g,b
	dev_rgbSOA.b = dev_rgbSOA.r + (pixCount);//Offset pointers for convenience
	dev_rgbSOA.g = dev_rgbSOA.b + (pixCount);//Offset pointers for convenience


	//Vertex Map Pyramid SOA. 
	cudaMalloc((void**) &dev_vmapSOA.x[0], sizeof(float)*3*(pixCount + pixCount>>2 + pixCount>>4));
	//Get convenience pointer offsets
	dev_vmapSOA.x[1] = dev_vmapSOA.x[0] + pixCount;
	dev_vmapSOA.x[2] = dev_vmapSOA.x[1] + pixCount >> 2;
	dev_vmapSOA.y[0] = dev_vmapSOA.x[2] + pixCount >> 4;
	dev_vmapSOA.y[1] = dev_vmapSOA.x[0] + pixCount;
	dev_vmapSOA.y[2] = dev_vmapSOA.y[1] + pixCount >> 2;
	dev_vmapSOA.z[0] = dev_vmapSOA.y[2] + pixCount >> 4;
	dev_vmapSOA.z[1] = dev_vmapSOA.z[0] + pixCount;
	dev_vmapSOA.z[2] = dev_vmapSOA.z[1] + pixCount >> 2;

	//Normal Map Pyramid SOA. 
	cudaMalloc((void**) &dev_nmapSOA.x[0], sizeof(float)*3*(pixCount + pixCount>>2 + pixCount>>4));
	//Get convenience pointer offsets
	dev_nmapSOA.x[1] = dev_nmapSOA.x[0] + pixCount;
	dev_nmapSOA.x[2] = dev_nmapSOA.x[1] + pixCount >> 2;
	dev_nmapSOA.y[0] = dev_nmapSOA.x[2] + pixCount >> 4;
	dev_nmapSOA.y[1] = dev_nmapSOA.x[0] + pixCount;
	dev_nmapSOA.y[2] = dev_nmapSOA.y[1] + pixCount >> 2;
	dev_nmapSOA.z[0] = dev_nmapSOA.y[2] + pixCount >> 4;
	dev_nmapSOA.z[1] = dev_nmapSOA.z[0] + pixCount;
	dev_nmapSOA.z[2] = dev_nmapSOA.z[1] + pixCount >> 2;



}

void MeshTracker::cleanupCuda()
{
	cudaFree(dev_colorImageBuffer);
	cudaFree(dev_depthImageBuffer);
	cudaFree(dev_rgbmapSOA);
	cudaFree(dev_vmapSOA);
	cudaFree(dev_nmapSOA);
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