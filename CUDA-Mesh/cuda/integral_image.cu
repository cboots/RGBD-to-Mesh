#include "integral_image.h"



__host__ void createIntegralImage(float* original, float* tempbuffer, float* integral, int width, int height)
{
	exclusiveScanRows(original, integral, width, height);
	transpose(integral, tempbuffer, width, height);
	exclusiveScanRows(tempbuffer, tempbuffer, height, width);
	transpose(tempbuffer, integral, height, width);
}


__device__ float AreaSum(float* integralImage, int imageWidth, int imageHeight, 
						int kernelLeft, int kernelRight, int kernelTop, int kernelBottom)
{
	//UL + LR - UR - LL
	int ul = kernelTop * imageWidth + kernelLeft;
	int lr = kernelBottom * imageWidth + kernelRight;
	int ur = kernelTop * imageWidth + kernelRight;
	int ll = kernelBottom * imageWidth + kernelLeft;
	return integralImage[ul] + integralImage[lr] + integralImage[ur] + integralImage[ll];
}