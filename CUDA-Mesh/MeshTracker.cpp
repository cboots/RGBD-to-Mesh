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


	host_normalX = new float[xRes*yRes];
	host_normalY = new float[xRes*yRes];

	cudaMalloc((void**) &dev_normalVoxels,	NUM_NORMAL_X_SUBDIVISIONS*NUM_NORMAL_Y_SUBDIVISIONS*sizeof(int));

	host_normalVoxels = new int[NUM_NORMAL_X_SUBDIVISIONS*NUM_NORMAL_Y_SUBDIVISIONS];

	//Decoupled voxels:
	createInt3SOA(dev_normalDecoupledHistogram, NUM_DECOUPLED_HISTOGRAM_BINS);
	createInt3SOA(dev_normalDecoupledHistogramPeaks, MAX_DECOUPLED_PEAKS);

	createInt3SOA(dev_normalSegments, xRes*yRes);

	createFloat3SOA(dev_normalPeaks, MAX_2D_PEAKS_PER_ROUND);

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

	delete host_normalX;
	delete host_normalY;

	cudaFree(dev_normalVoxels);
	delete host_normalVoxels;

	freeInt3SOA(dev_normalDecoupledHistogram);
	freeInt3SOA(dev_normalDecoupledHistogramPeaks);
	freeInt3SOA(dev_normalSegments);

	freeFloat3SOA(dev_normalPeaks);

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

void MeshTracker::createInt3SOA(Int3SOA& dev_soa, int length)
{
	cudaMalloc((void**) &dev_soa.x, sizeof(int)*3*length);
	dev_soa.y = dev_soa.x + length;
	dev_soa.z = dev_soa.y + length;
}

void MeshTracker::freeInt3SOA(Int3SOA dev_soa)
{
	cudaFree(dev_soa.x);
}


void MeshTracker::createFloat3SOA(Float3SOA& dev_soa, int length)
{
	cudaMalloc((void**) &dev_soa.x, sizeof(float)*3*length);
	dev_soa.y = dev_soa.x + length;
	dev_soa.z = dev_soa.y + length;
}

void MeshTracker::freeFloat3SOA(Float3SOA dev_soa)
{
	cudaFree(dev_soa.x);
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
	setSeperableKernelGaussian(10.0);

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
	for(int i = 0; i < NUM_NORMAL_X_SUBDIVISIONS*NUM_NORMAL_Y_SUBDIVISIONS; ++i)
		host_normalVoxels[i] = 0;


	int length = mXRes*mYRes;
	for(int i = 0; i < length; ++i)
	{
		float x = host_normalX[i];
		float y = host_normalY[i];
		if(x == x && y == y)//Will be false if NaN
		{
			int xI = (x+1.0f)*0.5f*NUM_NORMAL_X_SUBDIVISIONS;//x in range of -1 to 1. Map to 0 to 1.0 and multiply by number of bins
			int yI = (y+1.0f)*0.5f*NUM_NORMAL_Y_SUBDIVISIONS;//x in range of -1 to 1. Map to 0 to 1.0 and multiply by number of bins
			host_normalVoxels[xI + yI * NUM_NORMAL_X_SUBDIVISIONS]++;
		}
	}
}


void MeshTracker::GPUSimpleSegmentation()
{
	

	clearHistogram(dev_normalVoxels, NUM_NORMAL_X_SUBDIVISIONS, NUM_NORMAL_Y_SUBDIVISIONS);
	computeNormalHistogram(dev_nmapSOA.x[0], dev_nmapSOA.y[0], dev_normalVoxels, mXRes, mYRes, 
		NUM_NORMAL_X_SUBDIVISIONS, NUM_NORMAL_Y_SUBDIVISIONS);
	
	normalHistogramPrimaryPeakDetection(dev_normalVoxels, NUM_NORMAL_X_SUBDIVISIONS, NUM_NORMAL_Y_SUBDIVISIONS, 
		dev_normalPeaks, MAX_2D_PEAKS_PER_ROUND,  PEAK_2D_EXCLUSION_RADIUS, MIN_2D_PEAK_COUNT);
	
	Float3SOA normals;
	normals.x = dev_nmapSOA.x[0];
	normals.y = dev_nmapSOA.y[0];
	normals.z = dev_nmapSOA.z[0];
	segmentNormals2D(normals, dev_normalSegments, mXRes, mYRes, 
		dev_normalVoxels, NUM_NORMAL_X_SUBDIVISIONS, NUM_NORMAL_Y_SUBDIVISIONS, 
		dev_normalPeaks, MAX_2D_PEAKS_PER_ROUND, 10.0f*PI/180.0f);

	 //Debug
	Float3SOA peaksCopy;
	peaksCopy.x = new float[MAX_2D_PEAKS_PER_ROUND*3];
	peaksCopy.y = peaksCopy.x + MAX_2D_PEAKS_PER_ROUND;
	peaksCopy.z = peaksCopy.y + MAX_2D_PEAKS_PER_ROUND;

	cudaMemcpy(peaksCopy.x, dev_normalPeaks.x, MAX_2D_PEAKS_PER_ROUND*3*sizeof(float), cudaMemcpyDeviceToHost);

	cout << "X Peaks: ";
	for(int i = 0; i < MAX_2D_PEAKS_PER_ROUND; ++i){
	if(peaksCopy.x[i] >= 0)
	cout << peaksCopy.x[i] << ',';
	}
	cout << endl;

	cout << "Y Peaks: ";
	for(int i = 0; i < MAX_2D_PEAKS_PER_ROUND; ++i){
	if(peaksCopy.y[i] >= 0)
	cout << peaksCopy.y[i] << ',';
	}
	cout << endl;

	cout << "Z Peaks: ";
	for(int i = 0; i < MAX_2D_PEAKS_PER_ROUND; ++i){
	if(peaksCopy.z[i] >= 0)
	cout << peaksCopy.z[i] << ',';
	}
	cout << endl;


	delete peaksCopy.x;


}


void MeshTracker::GPUDecoupledSegmentation()
{
	clearHistogram(dev_normalDecoupledHistogram.x, NUM_DECOUPLED_HISTOGRAM_BINS, 1);
	clearHistogram(dev_normalDecoupledHistogram.y, NUM_DECOUPLED_HISTOGRAM_BINS, 1);
	//clearHistogram(dev_normalDecoupledHistogram.z, NUM_DECOUPLED_HISTOGRAM_BINS, 1);

	ACosHistogram(dev_nmapSOA.x[0], dev_normalDecoupledHistogram.x, mXRes*mYRes, NUM_DECOUPLED_HISTOGRAM_BINS);
	ACosHistogram(dev_nmapSOA.y[0], dev_normalDecoupledHistogram.y, mXRes*mYRes, NUM_DECOUPLED_HISTOGRAM_BINS);
	//ACosHistogram(dev_nmapSOA.z[0], dev_normalDecoupledHistogram.z, mXRes*mYRes, NUM_DECOUPLED_HISTOGRAM_BINS);


	gaussianSubtractionPeakDetection(dev_normalDecoupledHistogram, dev_normalDecoupledHistogramPeaks, 
		NUM_DECOUPLED_HISTOGRAM_BINS, MAX_DECOUPLED_PEAKS, MIN_DECOUPLED_PEAK_COUNT, glm::vec3(15,15,20));

	/*
	//Begin peak debug code
	Int3SOA peaksCopy;
	peaksCopy.x = new int[MAX_DECOUPLED_PEAKS*3];
	peaksCopy.y = peaksCopy.x + MAX_DECOUPLED_PEAKS;
	peaksCopy.z = peaksCopy.y + MAX_DECOUPLED_PEAKS;

	cudaMemcpy(peaksCopy.x, dev_normalDecoupledHistogramPeaks.x, MAX_DECOUPLED_PEAKS*3*sizeof(int), cudaMemcpyDeviceToHost);

	cout << "X Peaks: ";
	for(int i = 0; i < MAX_DECOUPLED_PEAKS; ++i){
	if(peaksCopy.x[i] >= 0)
	cout << peaksCopy.x[i] << ',';
	}
	cout << endl;

	cout << "Y Peaks: ";
	for(int i = 0; i < MAX_DECOUPLED_PEAKS; ++i){
	if(peaksCopy.y[i] >= 0)
	cout << peaksCopy.y[i] << ',';
	}
	cout << endl;

	cout << "Z Peaks: ";
	for(int i = 0; i < MAX_DECOUPLED_PEAKS; ++i){
	if(peaksCopy.z[i] >= 0)
	cout << peaksCopy.z[i] << ',';
	}
	cout << endl;


	delete peaksCopy.x;

	//END DEBUG PRINT
	*/

	Float3SOA normals;
	normals.x = dev_nmapSOA.x[0];
	normals.y = dev_nmapSOA.y[0];
	normals.z = dev_nmapSOA.z[0];

	segmentNormals(normals, dev_normalSegments, mXRes, mYRes, 
		dev_normalDecoupledHistogram, NUM_DECOUPLED_HISTOGRAM_BINS, 
		dev_normalDecoupledHistogramPeaks, MAX_DECOUPLED_PEAKS, MAX_PEAK_RANGE);
}

void MeshTracker::copyXYNormalsToHost()
{
	cudaMemcpy(host_normalX, dev_nmapSOA.x[0], mXRes*mYRes*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_normalY, dev_nmapSOA.y[0], mXRes*mYRes*sizeof(float), cudaMemcpyDeviceToHost);
}


void MeshTracker::copyNormalVoxelsToGPU()
{
	cudaMemcpy(dev_normalVoxels, host_normalVoxels, NUM_NORMAL_X_SUBDIVISIONS*NUM_NORMAL_Y_SUBDIVISIONS*sizeof(int), cudaMemcpyHostToDevice);
}

void MeshTracker::subsamplePyramids()
{
	subsamplePyramidCUDA(dev_vmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
	subsamplePyramidCUDA(dev_nmapSOA, mXRes, mYRes, NUM_PYRAMID_LEVELS);
	subsamplePyramidCUDA(dev_rgbSOA,  mXRes, mYRes, NUM_PYRAMID_LEVELS);
}


#pragma endregion

#pragma endregion