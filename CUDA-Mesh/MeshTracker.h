#pragma once
#include "RGBDFrame.h"
#include "device_structs.h"
#include "cuda_runtime.h"
#include "preprocessing.h"
#include "normal_estimates.h"
#include "integral_image.h"
#include "gradient.h"
#include "seperable_filter.h"
#include "Utils.h"
#include "plane_segmentation.h"
#include <iostream>

using namespace std;
using namespace rgbd::framework;

#define NUM_FLOAT1_IMAGE_SIZE_BUFFERS 10
#define NUM_FLOAT1_PYRAMID_BUFFERS 10
#define NUM_FLOAT3_PYRAMID_BUFFERS 5

#define NUM_NORMAL_X_SUBDIVISIONS		32
#define NUM_NORMAL_Y_SUBDIVISIONS		32

#define NUM_DECOUPLED_HISTOGRAM_BINS	256

#define MAX_DECOUPLED_PEAKS			8
#define MAX_PEAK_RANGE				20
#define MIN_DECOUPLED_PEAK_COUNT	550


#define MAX_2D_PEAKS_PER_ROUND		8
#define MIN_2D_PEAK_COUNT			500
#define PEAK_2D_EXCLUSION_RADIUS	2


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

	Float3SOAPyramid dev_rgbSOA;
	Float3SOAPyramid dev_vmapSOA;
	Float3SOAPyramid dev_nmapSOA;

	float* dev_curvature;

	
	float* host_curvature;
	float* host_normalX;
	float* host_normalY;

	int* host_normalVoxels;
	int* dev_normalVoxels;

	Int3SOA dev_normalDecoupledHistogram;
	Int3SOA dev_normalDecoupledHistogramPeaks;

	Int3SOA dev_normalSegments;
	Float3SOA dev_normalPeaks;

	Float3SOAPyramid dev_float3PyramidBuffers[NUM_FLOAT3_PYRAMID_BUFFERS];
	Float1SOAPyramid dev_float1PyramidBuffers[NUM_FLOAT1_PYRAMID_BUFFERS];

	float* dev_floatImageBuffers[NUM_FLOAT1_IMAGE_SIZE_BUFFERS];

#pragma region
	
#pragma region Private Methods
	void createFloat1SOAPyramid(Float1SOAPyramid& dev_pyramid, int xRes, int yRes);
	void freeFloat1SOAPyramid(Float1SOAPyramid dev_pyramid);

	void createFloat3SOAPyramid(Float3SOAPyramid& dev_pyramid, int xRes, int yRes);
	void freeFloat3SOAPyramid(Float3SOAPyramid dev_pyramid);

	void createFloat3SOA(Float3SOA& dev_soa, int length);
	void freeFloat3SOA(Float3SOA dev_soa);

	
	void createInt3SOA(Int3SOA& dev_soa, int length);
	void freeInt3SOA(Int3SOA dev_soa);


	void initBuffers(int xRes, int yResolution);
	void cleanupBuffers();
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

	void buildRGBSOA();

	void setGaussianSpatialSigma(float sigma);
	void buildVMapNoFilter(float maxDepth);
	void buildVMapGaussianFilter(float maxDepth);
	void buildVMapBilateralFilter(float maxDepth, float sigma_t);

	void buildNMapSimple();
	void buildNMapAverageGradient(int windowRadius);
	void buildNMapPCA(float radiusMeters);


	void estimateCurvatureFromNormals();
	void CPUSimpleSegmentation();
	void GPUSimpleSegmentation();
	void copyXYNormalsToHost();
	void copyNormalVoxelsToGPU();
	void subsamplePyramids();

	void GPUDecoupledSegmentation();
#pragma endregion

#pragma region Buffer getters
	inline ColorPixel* getColorImageDevicePtr() { return dev_colorImageBuffer;}
	inline DPixel* getDepthImageDevicePtr() { return dev_depthImageBuffer;}
	inline Float3SOAPyramid getVMapPyramid() { return dev_vmapSOA;}
	inline Float3SOAPyramid getNMapPyramid() { return dev_nmapSOA;}
	inline Float3SOAPyramid getRGBMapSOA() { return dev_rgbSOA;}
	inline float* getCurvature() {return dev_curvature;}
	inline int* getDeviceNormalHistogram() { return dev_normalVoxels;}
	inline Int3SOA getDecoupledHistogram() { return dev_normalDecoupledHistogram;}
	inline Int3SOA getNormalSegments() { return dev_normalSegments;}
#pragma endregion

#pragma region Property Getters
	inline int getNormalXSubdivisions() { return NUM_NORMAL_X_SUBDIVISIONS; }
	inline int getNormalYSubdivisions() { return NUM_NORMAL_Y_SUBDIVISIONS; }
	inline int getNormalDecoupledBins() { return NUM_DECOUPLED_HISTOGRAM_BINS; }
#pragma endregion
};

