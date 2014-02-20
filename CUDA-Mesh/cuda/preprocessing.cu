#include "preprocessing.h"

__global__ void buildVMapNoFilterKernel(rgbd::framework::DPixel* dev_depthBuffer, VMapSOA vmapSOA, int xRes, int yRes,
										rgbd::framework::Intrinsics intr, float maxDepth) 
{
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

__host__ void buildVMapNoFilterCUDA(rgbd::framework::DPixel* dev_depthBuffer, VMapSOA vmapSOA, int xRes, int yRes, 
									rgbd::framework::Intrinsics intr, float maxDepth)
{

	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	buildVMapNoFilterKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_depthBuffer, vmapSOA, xRes, yRes, intr, maxDepth);

}



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

__host__ void rgbAOSToSOACUDA(rgbd::framework::ColorPixel* dev_colorPixels, 
							  RGBMapSOA rgbSOA, int xRes, int yRes)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	rgbAOSToSOAKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_colorPixels, rgbSOA, xRes, yRes);
}


//Subsamples float3 SOA by 1/2 and stores in dest
//Threads are parallel by `
__global__ void subsampleVMAPKernel(float* x_src, float* y_src, float* z_src, 
									float* x_dest, float* y_dest, float* z_dest,
									int xRes_src, int yRes_src)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	int xRes_dest = xRes_src >> 1;
	int yRes_dest = yRes_src >> 1;

	if(u < xRes_dest  && v < yRes_dest) 
	{
		int i_src = (v<<1)*xRes_src+(u<<1);
		int i_dest = (v * xRes_dest) + u;

		x_dest[i_dest] = x_src[i_src];
		y_dest[i_dest] = y_src[i_src];
		z_dest[i_dest] = z_src[i_src];


	}
}

__host__ void buildVMapPyramidCUDA(VMapSOA dev_vmapSOA, int xRes, int yRes, int numLevels)
{
	int tileSize = 16;

	for(int i = 0; i < numLevels - 1; ++i)
	{
		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(xRes>>(1+i))/float(tileSize)), 
			(int)ceil(float(yRes>>(1+i))/float(tileSize)));


		subsampleVMAPKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(dev_vmapSOA.x[i], dev_vmapSOA.y[i], dev_vmapSOA.z[i],
			dev_vmapSOA.x[i+1], dev_vmapSOA.y[i+1], dev_vmapSOA.z[i+1],
			xRes>>i, yRes>>i);
	}

}