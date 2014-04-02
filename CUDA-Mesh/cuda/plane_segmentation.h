#pragma once

#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_structs.h"
#include "RGBDFrame.h"
#include "Calibration.h"
#include <glm/glm.hpp>
#include "Utils.h"
#include "math.h"

__host__ void computeNormalHistogram(float* normX, float* normY, int* histogram, int xRes, int yRes, int xBins, int yBins);
__host__ void clearHistogram(int* histogram, int xBins, int yBins);


__host__ void ACosHistogram(float* cosineValue, int* histogram, int valueCount, int numBins);

__host__ void gaussianSubtractionPeakDetection(Int3SOA decoupledHist, Int3SOA peakIndex, int histSize, int maxPeaks, int minPeakCount, glm::vec3 sigmas);


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