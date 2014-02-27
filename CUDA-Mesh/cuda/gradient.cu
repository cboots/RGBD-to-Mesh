#include "gradient.h"

__global__ void horizontalGradientKernel(float* image, float* grad,
										 int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){

		float diff = 0.0f;

		if(u < xRes - 1 && u > 0)
		{
			//Diff to right
			diff = image[i+1] - image[i-1];
		}

		grad[i] = isnan(diff)?0.0f:diff;

	}
}


__global__ void verticalGradientKernel(float* image, float* grad,
									   int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){

		float diff = 0.0f;

		if(v < yRes - 1 && v > 0)
		{
			//Diff to right
			diff = image[i+xRes] - image[i-xRes];
		}

		grad[i] = isnan(diff)?0.0f:diff;

	}
}




__host__ void horizontalGradient(float* image_in, float* gradient_out, int width, int height)
{
	int tileX = 16;
	int tileY = 16;

	dim3 threads(tileX,tileY);
	dim3 blocks((int)ceil(float(width)/float(tileX)), 
		(int)ceil(float(height)/float(tileY)));

	horizontalGradientKernel<<<blocks,threads>>>(image_in, gradient_out, width, height);
}

__host__ void verticalGradient(float* image_in, float* gradient_out, int width, int height)
{
	int tileX = 16;
	int tileY = 16;

	dim3 threads(tileX,tileY);
	dim3 blocks((int)ceil(float(width)/float(tileX)), 
		(int)ceil(float(height)/float(tileY)));

	verticalGradientKernel<<<blocks,threads>>>(image_in, gradient_out, width, height);
}
