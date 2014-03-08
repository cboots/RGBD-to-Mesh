#include "MeshTracker.h"

#pragma region Ctor/Dtor

MeshTracker::MeshTracker(int xResolution, int yResolution, Intrinsics intr)
{
	mXRes = xResolution;
	mYRes = yResolution;
	mIntr = intr;

	initBuffers(mXRes, mYRes);

	resetTracker();
}


MeshTracker::~MeshTracker(void)
{
	cleanupBuffers();
}
#pragma endregion

#pragma region Setup/Teardown functions
void MeshTracker::initBuffers(int xRes, int yRes)
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

	//Curvature
	cudaMalloc((void**) &dev_curvature,		xRes*yRes*sizeof(float));
	cudaMalloc((void**) &dev_azimuthAngle,	xRes*yRes*sizeof(float));
	cudaMalloc((void**) &dev_polarAngle,	xRes*yRes*sizeof(float));

	host_azimuthAngle = new float[xRes*yRes];
	host_polarAngle = new float[xRes*yRes];

	cudaMalloc((void**) &dev_normalVoxels,	NUM_AZIMUTH_SUBDIVISIONS*NUM_POLAR_SUBDIVISIONS*sizeof(int));

	host_normalVoxels = new int[NUM_AZIMUTH_SUBDIVISIONS*NUM_POLAR_SUBDIVISIONS];

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

void MeshTracker::cleanupBuffers()
{
	cudaFree(dev_colorImageBuffer);
	cudaFree(dev_depthImageBuffer);
	freeFloat3SOAPyramid(dev_rgbSOA);
	freeFloat3SOAPyramid(dev_vmapSOA);
	freeFloat3SOAPyramid(dev_nmapSOA);

	cudaFree(dev_curvature);
	cudaFree(dev_azimuthAngle);
	cudaFree(dev_polarAngle);

	delete host_azimuthAngle;
	delete host_polarAngle;

	cudaFree(dev_normalVoxels);
	delete host_normalVoxels;


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
}

void MeshTracker::buildVMapNoFilter(float maxDepth)
{
	buildVMapNoFilterCUDA(dev_depthImageBuffer, dev_vmapSOA, mXRes, mYRes, mIntr, maxDepth);

}

void MeshTracker::buildVMapGaussianFilter(float maxDepth)
{
	buildVMapGaussianFilterCUDA(dev_depthImageBuffer, dev_vmapSOA, mXRes, mYRes, mIntr, maxDepth);

}

void MeshTracker::buildVMapBilateralFilter(float maxDepth, float sigma_t)
{

	buildVMapBilateralFilterCUDA(dev_depthImageBuffer, dev_vmapSOA, mXRes, mYRes, 
		mIntr, maxDepth, sigma_t);


}


void MeshTracker::setGaussianSpatialSigma(float sigma)
{
	setGaussianSpatialKernel(sigma);
}


void MeshTracker::buildNMapSimple()
{

	simpleNormals(dev_vmapSOA, dev_nmapSOA, NUM_PYRAMID_LEVELS, mXRes, mYRes);

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

	computeAverageGradientNormals(dev_float3PyramidBuffers[0], dev_float3PyramidBuffers[1], dev_nmapSOA, mXRes, mYRes);
	
}

void MeshTracker::buildNMapPCA(float radiusMeters)
{
	computePCANormals(dev_vmapSOA, dev_nmapSOA, dev_curvature, mXRes, mYRes, radiusMeters);	
}

void MeshTracker::estimateCurvatureFromNormals()
{
	curvatureEstimate(dev_nmapSOA, dev_curvature, mXRes, mYRes);
}


void MeshTracker::CPUSimpleSegmentation()
{
	//clear
	for(int i = 0; i < NUM_AZIMUTH_SUBDIVISIONS*NUM_POLAR_SUBDIVISIONS; ++i)
		host_normalVoxels[i] = 0;


	int length = mXRes*mYRes;
	for(int i = 0; i < length; ++i)
	{
		float azimuth = host_azimuthAngle[i];
		float polar = host_polarAngle[i];
		if(polar == polar && azimuth == azimuth && polar > 0.0f && azimuth > 0.0f)//Will be false if nan
			host_normalVoxels[POLAR_INDEX(polar)*NUM_AZIMUTH_SUBDIVISIONS + AZIMUTH_INDEX(azimuth)]++;
	}
}

void MeshTracker::GPUSimpleSegmentation()
{
	clearHistogram(dev_normalVoxels, NUM_AZIMUTH_SUBDIVISIONS, NUM_POLAR_SUBDIVISIONS);
	computeNormalHistogram(dev_azimuthAngle, dev_polarAngle, dev_normalVoxels, mXRes, mYRes, NUM_AZIMUTH_SUBDIVISIONS, NUM_POLAR_SUBDIVISIONS);
}

void MeshTracker::copySphericalNormalsToHost()
{
	cudaMemcpy(host_azimuthAngle, dev_azimuthAngle, mXRes*mYRes*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_polarAngle,   dev_polarAngle, mXRes*mYRes*sizeof(float), cudaMemcpyDeviceToHost);
}


void MeshTracker::copyNormalVoxelsToGPU()
{
	cudaMemcpy(dev_normalVoxels, host_normalVoxels,NUM_AZIMUTH_SUBDIVISIONS*NUM_POLAR_SUBDIVISIONS*sizeof(int), cudaMemcpyHostToDevice);
}

void MeshTracker::generateSphericalNormals()
{
	convertNormalToSpherical(dev_nmapSOA.x[0], dev_nmapSOA.y[0], dev_nmapSOA.z[0], dev_azimuthAngle, dev_polarAngle, mXRes*mYRes);
}

void MeshTracker::subsamplePyramids()
{
	subsamplePyramidCUDA(dev_vmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
	subsamplePyramidCUDA(dev_nmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
	subsamplePyramidCUDA(dev_rgbSOA,  mXRes, mYRes, NUM_PYRAMID_LEVELS);
}


#pragma endregion

#pragma endregion