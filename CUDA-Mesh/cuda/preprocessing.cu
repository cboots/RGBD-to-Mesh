#include "preprocessing.h"

__global__ void buildVMapNoFilterKernel(rgbd::framework::DPixel* dev_depthBuffer, VMapSOA vmapSOA, int xRes, int yRes,
										rgbd::framework::Intrinsics intr, float maxDepth) {
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if(u < xRes && v < yRes) 
	{
		int i = (v * xRes) + u;

		float x = 0.0f;
		float y = 0.0f;
		float z = dev_depthBuffer[i].depth * 0.001f;

		if (z > 0.001f && z < maxDepth) {//Exclude zero or negative depths.  
			
			x = (u - intr.cx) * z / intr.fx;
			y = (v - intr.cy) * z / intr.fy;
		} 

		//Write to SOA in memory coallesed way
		vmapSOA.x[0][i] = x;
		vmapSOA.y[0][i] = y;
		vmapSOA.z[0][i] = z;
	}
}

__host__ void buildVMapNoFilter(rgbd::framework::DPixel* dev_depthBuffer, VMapSOA vmapSOA, int xRes, int yRes, 
								rgbd::framework::Intrinsics intr, float maxDepth)
{

	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	buildVMapNoFilterKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_depthBuffer, vmapSOA, xRes, yRes, intr, maxDepth);

}


/*
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
*/

__global__ void rgbAOSToSOAKernel(rgbd::framework::ColorPixel* dev_colorPixels, 
						  RGBMapSOA rgbSOA, int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if(u < xRes && v < yRes) 
	{
		int i = (v * xRes) + u;

		rgbd::framework::ColorPixel color = dev_colorPixels[i];
		rgbSOA.r[i] = color.r / 255.0f;
		rgbSOA.g[i] = color.g / 255.0f;
		rgbSOA.b[i] = color.b / 255.0f;
	}

}

__host__ void rgbAOSToSOA(rgbd::framework::ColorPixel* dev_colorPixels, 
						  RGBMapSOA rgbSOA, int xRes, int yRes)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	rgbAOSToSOAKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_colorPixels, rgbSOA, xRes, yRes);
}
