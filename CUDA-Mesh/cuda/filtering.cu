#include "filtering.h"

__global__ void depthBufferToFloatKernel(rgbd::framework::DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes) {
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * xRes) + c;

	if(r < yRes && c < xRes) {
		dev_depthFloat[i] = dev_depthBuffer[i].depth*0.001f; // depth converted from mm to meters
	}
}

__host__ void depthBufferToFloat(rgbd::framework::DPixel* dev_depthBuffer, float* dev_depthFloat, int xRes, int yRes)
{

	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	depthBufferToFloatKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_depthBuffer, dev_depthFloat, xRes, yRes);

}



__global__ void makePointCloudKernel(float* dev_depthBuffer, rgbd::framework::ColorPixel* dev_colorPixels, 
							   PointCloud* dev_pointCloudBuffer,
								  int xRes, int yRes, rgbd::framework::Intrinsics intr, float maxDepth)
{
	int u = (blockIdx.y * blockDim.y) + threadIdx.y;
	int v = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (u * xRes) + v;

	if(u < yRes && v < xRes) 
	{
		// In range
		PointCloud point;
		point.pos.z = dev_depthBuffer[i];
		if (point.pos.z > 0.001f && point.pos.z < maxDepth) {//Exclude zero or negative depths.  
			rgbd::framework::ColorPixel cPixel = dev_colorPixels[i];
			point.color.r = cPixel.r/255.0f;
			point.color.g = cPixel.g/255.0f;
			point.color.b = cPixel.b/255.0f;

			point.pos.x = (u - intr.cx) * point.pos.z / intr.fx;
			point.pos.y = (v - intr.cy) * point.pos.z / intr.fy;
		} 

		dev_pointCloudBuffer[i] = point;
	}
}

__host__ void convertToPointCloud(float* dev_depthBuffer, rgbd::framework::ColorPixel* dev_colorPixels, 
								  PointCloud* dev_pointCloudBuffer,
								  int xRes, int yRes, rgbd::framework::Intrinsics intr, float maxDepth)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	makePointCloudKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_depthBuffer, dev_colorPixels, dev_pointCloudBuffer,
		xRes, yRes, intr, maxDepth);
}