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
#include "CudaUtils.h"
#include "plane_segmentation.h"
#include "quadtree.h"
#include <iostream>

using namespace std;
using namespace rgbd::framework;

#define NUM_FLOAT1_IMAGE_SIZE_BUFFERS 1
#define NUM_FLOAT1_PYRAMID_BUFFERS 1
#define NUM_FLOAT3_PYRAMID_BUFFERS 5

#define NUM_NORMAL_X_SUBDIVISIONS		32
#define NUM_NORMAL_Y_SUBDIVISIONS		32

#define MAX_2D_PEAKS_PER_ROUND		4
#define PEAK_2D_EXCLUSION_RADIUS	8

#define MAX_SEGMENTATION_ROUNDS		2
#define MAX_PLANES_TOTAL			(MAX_2D_PEAKS_PER_ROUND*DISTANCE_HIST_MAX_PEAKS*MAX_SEGMENTATION_ROUNDS)

#define DISTANCE_HIST_MAX_PEAKS	8
#define DISTANCE_HIST_COUNT	512
#define DISTANCE_HIST_MIN	0.1f
#define DISTANCE_HIST_MAX	5.0f
#define DISTANCE_HIST_RESOLUTION  ((DISTANCE_HIST_MAX-DISTANCE_HIST_MIN)/DISTANCE_HIST_COUNT)

#define MAX_TEXTURE_BUFFER_SIZE	1024

//For now, use theoretical max size. Should be able to decimate this considerably
#define QUADTREE_BUFFER_SIZE	(MAX_TEXTURE_BUFFER_SIZE*MAX_TEXTURE_BUFFER_SIZE)

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
	float m2DSegmentationMaxAngleFromPeak;

	float mDistPeakThresholdTight;
	float mMinDistPeakCount;
	float mPlaneMergeAngleThresh;
	float mPlaneMergeDistThresh;
	float mPlaneFinalAngleThresh;
	float mPlaneFinalDistThresh;
	float mMinNormalPeakCout;
	int mMaxPlanesOutput;
#pragma region

#pragma region Pipeline Buffer Device Pointers
	//PIPELINE BUFFERS
	ColorPixel* dev_colorImageBuffer;
	DPixel* dev_depthImageBuffer;

	Float3SOAPyramid dev_rgbSOA;
	Float3SOAPyramid dev_vmapSOA;
	Float3SOAPyramid dev_nmapSOA;

	//Segmentation buffers
	int* dev_normalVoxels;//2D Normal Histogram
	Float3SOA dev_normalPeaks;//Normal Peaks Array
	int* dev_normalSegments;//Normal Segmentation Buffer
	float* dev_planeProjectedDistanceMap;//Projetd Distance Buffer

	int* dev_distanceHistograms[MAX_2D_PEAKS_PER_ROUND];
	float* dev_distPeaks[MAX_2D_PEAKS_PER_ROUND];

	PlaneStats dev_planeStats;

	int* dev_finalSegmentsBuffer;
	float* dev_finalDistanceToPlaneBuffer;

	int* dev_planeIdMap;
	int* dev_planeInvIdMap;
	int* dev_detectedPlaneCount;
	int host_detectedPlaneCount;
	glm::vec3* dev_planeTangents;
	glm::vec4* dev_aabbIntermediateBuffer;
	glm::vec4* dev_planeAABB;

	float* dev_segmentProjectedSx;
	float* dev_segmentProjectedSy;

	ProjectionParameters* dev_planeProjectionParameters;
	ProjectionParameters* host_planeProjectionParameters;

	Float4SOA dev_PlaneTexture;
	int* dev_quadTreeAssembly;
	int* dev_quadTreeScanResults;
	int* dev_quadTreeBlockResults;
	
	//Quadtree mesh output
	int* dev_quadTreeIndexBuffer;
	float4* dev_quadTreeVertexBuffer;
	int* dev_compactCount;
	int host_quadtreeVertexCount;

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

	void createFloat4SOA(Float4SOA& dev_soa, int length);
	void freeFloat4SOA(Float4SOA dev_soa);

	void createInt3SOA(Int3SOA& dev_soa, int length);
	void freeInt3SOA(Int3SOA dev_soa);


	void initBuffers(int xRes, int yResolution);
	void cleanupBuffers();

	void segmentationInnerLoop(int resolutionLevel, int iteration);
	void normalHistogramGeneration(int normalHistLevel, int iteration);
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
	void buildNMapAverageGradient();

	void GPUSimpleSegmentation();
	void subsamplePyramids();

	void ReprojectPlaneTextures();
	
#pragma endregion

#pragma region Buffer getters
	inline ColorPixel* getColorImageDevicePtr() { return dev_colorImageBuffer;}
	inline DPixel* getDepthImageDevicePtr() { return dev_depthImageBuffer;}
	inline Float3SOAPyramid getVMapPyramid() { return dev_vmapSOA;}
	inline Float3SOAPyramid getNMapPyramid() { return dev_nmapSOA;}
	inline Float3SOAPyramid getRGBMapSOA() { return dev_rgbSOA;}
	inline int* getDeviceNormalHistogram() { return dev_normalVoxels;}
	inline int* getNormalSegments() { return dev_normalSegments;}
	inline float* getPlaneProjectedDistance() {return dev_planeProjectedDistanceMap;}
	inline int* getFinalSegments() { return dev_finalSegmentsBuffer;}
	inline float* getFinalFitDistance() {return dev_finalDistanceToPlaneBuffer;}
	inline float* getProjectedSx() {return dev_segmentProjectedSx;}
	inline float* getProjectedSy() {return dev_segmentProjectedSy;}
	inline int* getDistanceHistogram(int peak) {return (peak >= 0 && peak < MAX_2D_PEAKS_PER_ROUND)?dev_distanceHistograms[peak]:NULL;}
	inline Float4SOA getProjectedTexture(int planeNum){return dev_PlaneTexture;}
	inline ProjectionParameters getHostProjectionParameters(int planeNum){return host_planeProjectionParameters[planeNum];}
	inline int getHostNumDetectedPlanes(){return host_detectedPlaneCount;}
	inline int* getQuadtreeBuffer(int planeNum){return dev_quadTreeAssembly;}
#pragma endregion

#pragma region Property Getters
	inline int getNormalXSubdivisions() { return NUM_NORMAL_X_SUBDIVISIONS; }
	inline int getNormalYSubdivisions() { return NUM_NORMAL_Y_SUBDIVISIONS; }
	inline int getDistanceHistogramSize() {return DISTANCE_HIST_COUNT; }
	//In degrees
	inline float get2DSegmentationMaxAngle(){return m2DSegmentationMaxAngleFromPeak;}
	inline void set2DSegmentationMaxAngle(float maxAngleDegrees){
		if(maxAngleDegrees > 0.0f && maxAngleDegrees < 90.0f) 
			m2DSegmentationMaxAngleFromPeak = maxAngleDegrees;
	}
	inline int getProjectedTextureBufferWidth(){return MAX_TEXTURE_BUFFER_SIZE;}
	inline int getMaxPlanesOutput(){return mMaxPlanesOutput;}
	inline void setMaxPlanesOutput(int maxPlanes){if(maxPlanes > 0 && maxPlanes <= MAX_PLANES_TOTAL) mMaxPlanesOutput = maxPlanes;}
#pragma endregion
};

