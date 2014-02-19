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
		int i_dest = (v * xRes_dest) + u;
		int i_src_offset = u*xRes_src+v;

		int validCount = 0;

		//Load 4 pixels, UL, UR, LL, LR
		float x_accum = 0.0f;
		float y_accum = 0.0f;
		float z_accum = 0.0f;

		//====Upper left====
		float z = z_src[i_src_offset];//UL
		//Depth test
		if(z > 0.001f)
		{
			//Non-zero depth, proceed with other loads
			++validCount;
			x_accum += x_src[i_src_offset];
			y_accum += y_src[i_src_offset];
			z_accum += z;
		}

		//====Upper right ====
		z = z_src[i_src_offset+1];//UR
		//Depth test
		if(z > 0.001f)
		{
			//Non-zero depth, proceed with other loads
			++validCount;
			x_accum += x_src[i_src_offset+1];
			y_accum += y_src[i_src_offset+1];
			z_accum += z;
		}


		//====Lower left====
		z = z_src[i_src_offset+xRes_src];//LL
		//Depth test
		if(z > 0.001f)
		{
			//Non-zero depth, proceed with other loads
			++validCount;
			x_accum += x_src[i_src_offset+xRes_src];
			y_accum += y_src[i_src_offset+xRes_src];
			z_accum += z;
		}

		//====Lower right====
		z = z_src[i_src_offset+xRes_src + 1];//LR
		//Depth test
		if(z > 0.001f)
		{
			//Non-zero depth, proceed with other loads
			++validCount;
			x_accum += x_src[i_src_offset+xRes_src + 1];
			y_accum += y_src[i_src_offset+xRes_src + 1];
			z_accum += z;
		}


		//We have all subpixels accumulated now. Do average if validCount non-zero
		if(validCount > 0){
			x_dest[i_dest] = x_accum/validCount;
			y_dest[i_dest] = y_accum/validCount;
			z_dest[i_dest] = z_accum/validCount;
		}else{
			x_dest[i_dest] = 0.0f;
			y_dest[i_dest] = 0.0f;
			z_dest[i_dest] = 0.0f;
		}

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
			xRes, yRes);
	}

}