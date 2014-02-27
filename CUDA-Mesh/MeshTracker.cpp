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
	//RGB Pyramid SOA
	createFloat3SOAPyramid(dev_rgbSOA, xRes, yRes);

	//Vertex Map Pyramid SOA. 
	createFloat3SOAPyramid(dev_vmapSOA, xRes, yRes);

	//Normal Map Pyramid SOA. 
	createFloat3SOAPyramid(dev_nmapSOA, xRes, yRes);

	//Curvature pyramid SOA
	createFloat1SOAPyramid(dev_curvature, xRes, yRes);

	for(int i = 0; i < NUM_FLOAT1_PYRAMID_BUFFERS; ++i)
	{
		createFloat1SOAPyramid(dev_float1PyramidBuffers[i], xRes, yRes);
	}

	for(int i = 0; i < NUM_FLOAT3_PYRAMID_BUFFERS; ++i)
	{
		createFloat3SOAPyramid(dev_float3PyramidBuffers[i], xRes, yRes);
	}

	for(int i = 0; i < NUM_FLOAT1_IMAGE_SIZE_BUFFERS; ++i)
	{
		cudaMalloc((void**) &dev_floatImageBuffers[i], xRes*yRes*sizeof(float));
	}

	//Initialize gaussian spatial kernel
	setGaussianSpatialSigma(1.0f);
}

void MeshTracker::cleanupCuda()
{
	cudaFree(dev_colorImageBuffer);
	cudaFree(dev_depthImageBuffer);
	freeFloat3SOAPyramid(dev_rgbSOA);
	freeFloat3SOAPyramid(dev_vmapSOA);
	freeFloat3SOAPyramid(dev_nmapSOA);
	freeFloat1SOAPyramid(dev_curvature);

	for(int i = 0; i < NUM_FLOAT1_PYRAMID_BUFFERS; ++i)
	{
		freeFloat1SOAPyramid(dev_float1PyramidBuffers[i]);
	}

	for(int i = 0; i < NUM_FLOAT3_PYRAMID_BUFFERS; ++i)
	{
		freeFloat3SOAPyramid(dev_float3PyramidBuffers[i]);
	}

	for(int i = 0; i < NUM_FLOAT1_IMAGE_SIZE_BUFFERS; ++i)
	{
		cudaFree(dev_floatImageBuffers[i]);
	}


}


void MeshTracker::createFloat1SOAPyramid(Float1SOAPyramid& dev_pyramid, int xRes, int yRes)
{
	int pixCount = xRes*yRes;
	int pyramidCount = 0;

	for(int i = 0; i < NUM_PYRAMID_LEVELS; ++i)
	{
		pyramidCount += (pixCount >> (i*2));
	}

	cudaMalloc((void**) &dev_pyramid.x[0], sizeof(float)*(pyramidCount));
	//Get convenience pointer offsets
	for(int i = 0; i < NUM_PYRAMID_LEVELS-1; ++i)
	{
		dev_pyramid.x[i+1] = dev_pyramid.x[i] + (pixCount >> (i*2));
	}


}

void MeshTracker::freeFloat1SOAPyramid(Float1SOAPyramid dev_pyramid)
{
	cudaFree(dev_pyramid.x[0]);
}




void MeshTracker::createFloat3SOAPyramid(Float3SOAPyramid& dev_pyramid, int xRes, int yRes)
{
	int pixCount = xRes*yRes;
	int pyramidCount = 0;

	for(int i = 0; i < NUM_PYRAMID_LEVELS; ++i)
	{
		pyramidCount += (pixCount >> (i*2));
	}

	cudaMalloc((void**) &dev_pyramid.x[0], sizeof(float)*3*(pyramidCount));
	//Get convenience pointer offsets
	for(int i = 0; i < NUM_PYRAMID_LEVELS-1; ++i)
	{
		dev_pyramid.x[i+1] = dev_pyramid.x[i] + (pixCount >> (i*2));
	}

	dev_pyramid.y[0] = dev_pyramid.x[0] + pyramidCount;
	for(int i = 0; i < NUM_PYRAMID_LEVELS-1; ++i)
	{
		dev_pyramid.y[i+1] = dev_pyramid.y[i] + (pixCount >> (i*2));
	}


	dev_pyramid.z[0] = dev_pyramid.y[0] + pyramidCount;
	for(int i = 0; i < NUM_PYRAMID_LEVELS-1; ++i)
	{
		dev_pyramid.z[i+1] = dev_pyramid.z[i] + (pixCount >> (i*2));
	}

}

void MeshTracker::freeFloat3SOAPyramid(Float3SOAPyramid dev_pyramid)
{
	cudaFree(dev_pyramid.x[0]);
}

#pragma endregion

#pragma region Pipeline control API
void MeshTracker::resetTracker()
{
	lastFrameTime = 0LL;
	currentFrameTime = 0LL;
	//TODO: Initalize and clear world tree


}

#pragma region Preprocessing
void MeshTracker::pushRGBDFrameToDevice(ColorPixelArray colorArray, DPixelArray depthArray, timestamp time)
{
	lastFrameTime = currentFrameTime;
	currentFrameTime = time;

	cudaMemcpy((void*)dev_depthImageBuffer, depthArray.get(), sizeof(DPixel)*mXRes*mYRes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)dev_colorImageBuffer, colorArray.get(), sizeof(ColorPixel)*mXRes*mYRes, cudaMemcpyHostToDevice);

}



void MeshTracker::buildRGBSOA()
{
	rgbAOSToSOACUDA(dev_colorImageBuffer, dev_rgbSOA, mXRes, mYRes);
	buildRGBMapPyramid(dev_rgbSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
}

void MeshTracker::buildVMapNoFilter(float maxDepth)
{
	buildVMapNoFilterCUDA(dev_depthImageBuffer, dev_vmapSOA, mXRes, mYRes, mIntr, maxDepth);

	subsamplePyramidCUDA(dev_vmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
}

void MeshTracker::buildVMapGaussianFilter(float maxDepth)
{
	buildVMapGaussianFilterCUDA(dev_depthImageBuffer, dev_vmapSOA, mXRes, mYRes, mIntr, maxDepth);

	subsamplePyramidCUDA(dev_vmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
}

void MeshTracker::buildVMapBilateralFilter(float maxDepth, float sigma_t)
{

	buildVMapBilateralFilterCUDA(dev_depthImageBuffer, dev_vmapSOA, mXRes, mYRes, 
		mIntr, maxDepth, sigma_t);


	subsamplePyramidCUDA(dev_vmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
}


void MeshTracker::setGaussianSpatialSigma(float sigma)
{
	setGaussianSpatialKernel(sigma);
}


void MeshTracker::buildNMapSimple()
{

	simpleNormals(dev_vmapSOA, dev_nmapSOA, dev_curvature, NUM_PYRAMID_LEVELS, mXRes, mYRes);

}


void MeshTracker::buildNMapAverageGradient(int windowRadius)
{
	//Assemble gradient images.

	//For each first level of pyramid
	horizontalGradient(dev_vmapSOA.x[0], dev_float3PyramidBuffers[0].x[0], mXRes, mYRes);
	verticalGradient(  dev_vmapSOA.x[0], dev_float3PyramidBuffers[1].x[0], mXRes, mYRes);

	horizontalGradient(dev_vmapSOA.y[0], dev_float3PyramidBuffers[0].y[0], mXRes, mYRes);
	verticalGradient(  dev_vmapSOA.y[0], dev_float3PyramidBuffers[1].y[0], mXRes, mYRes);

	horizontalGradient(dev_vmapSOA.z[0], dev_float3PyramidBuffers[0].z[0], mXRes, mYRes);
	verticalGradient(  dev_vmapSOA.z[0], dev_float3PyramidBuffers[1].z[0], mXRes, mYRes);

	//Apply filters to float3
	//setSeperableKernelUniform();
	setSeperableKernelGaussian(2.0);

	seperableFilter(dev_float3PyramidBuffers[0].x[0], dev_float3PyramidBuffers[0].y[0], dev_float3PyramidBuffers[0].z[0],
		dev_float3PyramidBuffers[0].x[0], dev_float3PyramidBuffers[0].y[0], dev_float3PyramidBuffers[0].z[0],
		mXRes, mYRes);

	seperableFilter(dev_float3PyramidBuffers[1].x[0], dev_float3PyramidBuffers[1].y[0], dev_float3PyramidBuffers[1].z[0],
		dev_float3PyramidBuffers[1].x[0], dev_float3PyramidBuffers[1].y[0], dev_float3PyramidBuffers[1].z[0],
		mXRes, mYRes);

	computeAverageGradientNormals(dev_float3PyramidBuffers[0], dev_float3PyramidBuffers[1], dev_nmapSOA, dev_curvature, mXRes, mYRes);

	subsamplePyramidCUDA(dev_nmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
}

#pragma endregion

#pragma endregion