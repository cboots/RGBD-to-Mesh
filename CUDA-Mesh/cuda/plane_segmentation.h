#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include "Calibration.h"
#include <glm/glm.hpp>
#include "Utils.h"
#include "math.h"

__host__ void computeNormalHistogram(float* normX, float* normY, float* normZ, int* finalSegmentsBuffer, int* histogram, 
									 int xRes, int yRes, int xBins, int yBins, bool excludePreviousSegments);

__host__ void clearHistogram(int* histogram, int xBins, int yBins);


__host__ void ACosHistogram(float* cosineValue, int* histogram, int valueCount, int numBins);


__host__ void normalHistogramPrimaryPeakDetection(int* histogram, int xBins, int yBins, Float3SOA peaks, int maxPeaks, 
												  int exclusionRadius, int minPeakHeight);


__host__ void segmentNormals2D(Float3SOA rawNormals, Float3SOA rawPositions,
							   int* normalSegments,  float* projectedDistance, int imageWidth, int imageHeight, 
							   int* histogram, int xBins, int yBins, 
							   Float3SOA peaks, int maxPeaks, float maxAngleRange);

__host__ void generateDistanceHistograms(int* dev_normalSegments, float* dev_planeProjectedDistanceMap, int xRes, int yRes,
										 int** dev_distanceHistograms, int numMaxNormalSegments, 
										 int histcount, float histMinDist, float histMaxDist);


__host__ void distanceHistogramPrimaryPeakDetection(int* histogram, int length, int numHistograms, float* distPeaks, int maxDistPeaks, 
												  int exclusionRadius, int minPeakHeight, float minHistDist, float maxHistDist);

__host__ void fineDistanceSegmentation(float* distPeaks, int numNormalPeaks,  int maxDistPeaks, 
									   Float3SOA positions, PlaneStats planeStats,
									   int* normalSegments, float* planeProjectedDistanceMap, 
									   int xRes, int yRes, float maxDistTolerance, int iteration);


__host__ void clearPlaneStats(PlaneStats planeStats, int numNormalPeaks, int numDistPeaks, int maxRounds, int iteration);

__host__ void finalizePlanes(PlaneStats planeStats, int numNormalPeaks, int numDistPeaks, 
							 float mergeAngleThresh, float mergeDistThresh,  int iteration);

__host__ void fitFinalPlanes(PlaneStats planeStats, int numPlanes, 
							  Float3SOA norms, Float3SOA positions, int* finalSegmentsBuffer, float* distToPlaneBuffer, int xRes, int yRes,
							 float fitAngleThresh, float fitDistThresh, int iteration);

__host__ void realignPeaks(PlaneStats planeStats, Float3SOA normalPeaks, int numNormPeaks, int numDistPeaks, int xBins, int yBins, int iteration);