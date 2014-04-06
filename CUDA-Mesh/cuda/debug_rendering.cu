#include "debug_rendering.h"


// Kernel that writes the depth image to the OpenGL PBO directly.
__global__ void sendDepthImageBufferToPBO(float4* PBOpos, glm::vec2 resolution, DPixel* depthBuffer){

	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * resolution.x) + c;

	if(r<resolution.y && c<resolution.x) {

		// Cast to float for storage
		float depth = depthBuffer[i].depth;

		// Each thread writes one pixel location in the texture (textel)
		// Store depth in every component except alpha
		PBOpos[i].x = depth;
		PBOpos[i].y = depth;
		PBOpos[i].z = depth;
		PBOpos[i].w = 1.0f;
	}
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendColorImageBufferToPBO(float4* PBOpos, glm::vec2 resolution, ColorPixel* colorBuffer){

	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * resolution.x) + c;

	if(r<resolution.y && c<resolution.x){

		glm::vec3 color;
		color.r = colorBuffer[i].r/255.0f;
		color.g = colorBuffer[i].g/255.0f;
		color.b = colorBuffer[i].b/255.0f;

		// Each thread writes one pixel location in the texture (textel)
		PBOpos[i].x = color.r;
		PBOpos[i].y = color.g;
		PBOpos[i].z = color.b;
		PBOpos[i].w = 1.0f;
	}
}


// Draws depth image buffer to the texture.
// Texture width and height must match the resolution of the depth image.
// Returns false if width or height does not match, true otherwise
__host__ void drawDepthImageBufferToPBO(float4* dev_PBOpos, DPixel* dev_depthImageBuffer, int texWidth, int texHeight)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(texWidth)/float(tileSize)), 
		(int)ceil(float(texHeight)/float(tileSize)));

	sendDepthImageBufferToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_PBOpos, glm::vec2(texWidth, texHeight), dev_depthImageBuffer);

}

// Draws color image buffer to the texture.
// Texture width and height must match the resolution of the color image.
// Returns false if width or height does not match, true otherwise
// dev_PBOpos must be a CUDA device pointer
__host__ void drawColorImageBufferToPBO(float4* dev_PBOpos, ColorPixel* dev_colorImageBuffer, int texWidth, int texHeight)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(texWidth)/float(tileSize)), 
		(int)ceil(float(texHeight)/float(tileSize)));

	sendColorImageBufferToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_PBOpos, glm::vec2(texWidth, texHeight), dev_colorImageBuffer);

}


__global__ void sendFloat3SOAToPBO(float4* pbo, float* x_src, float* y_src, float* z_src, float w,
								   int xRes, int yRes, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * xRes) + x;
	int pboi = (y * pboXRes) + x;

	if(y < yRes && x < xRes){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = x_src[i];
		pbo[pboi].y = y_src[i];
		pbo[pboi].z = z_src[i];
		pbo[pboi].w = w;
	}
}


__global__ void sendFloat1SOAToPBO(float4* pbo, float* x_src, float w,
								   int xRes, int yRes, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * xRes) + x;
	int pboi = (y * pboXRes) + x;

	if(y < yRes && x < xRes){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = x_src[i];
		pbo[pboi].y = w;
		pbo[pboi].z = w;
		pbo[pboi].w = w;
	}
}


__global__ void clearPBOKernel(float4* pbo, int xRes, int yRes, float clearValue)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * xRes) + x;

	if(y < yRes && x < xRes){

		// Each thread writes one pixel location in the texture (textel)
		// 
		float4 clear;
		clear.x = clearValue;
		clear.y = clearValue;
		clear.z = clearValue;
		clear.w = clearValue;
		pbo[i] = clear;
	}
}


__host__ void clearPBO(float4* pbo, int xRes, int yRes, float clearValue)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	clearPBOKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo, xRes, yRes, clearValue);

}

__host__ void drawVMaptoPBO(float4* pbo, Float3SOAPyramid vmap, int level, int xRes, int yRes)
{
	int tileSize = 16;

	if(level < NUM_PYRAMID_LEVELS)
	{
		int scaledXRes = xRes >> level;
		int scaledYRes = yRes >> level;

		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(scaledXRes)/float(tileSize)), 
			(int)ceil(float(scaledYRes)/float(tileSize)));


		sendFloat3SOAToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo, vmap.x[level], vmap.y[level], vmap.z[level],  1.0,
			scaledXRes, scaledYRes, xRes, yRes);
	}
}


__host__ void drawNMaptoPBO(float4* pbo, Float3SOAPyramid nmap, int level, int xRes, int yRes)
{
	int tileSize = 16;

	if(level < NUM_PYRAMID_LEVELS)
	{
		int scaledXRes = xRes >> level;
		int scaledYRes = yRes >> level;

		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(scaledXRes)/float(tileSize)), 
			(int)ceil(float(scaledYRes)/float(tileSize)));


		sendFloat3SOAToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo, nmap.x[level], nmap.y[level], nmap.z[level],  0.0,
			scaledXRes, scaledYRes, xRes, yRes);
	}
}



__host__ void drawRGBMaptoPBO(float4* pbo, Float3SOAPyramid rgbMap, int level, int xRes, int yRes)
{
	int tileSize = 16;

	if(level < NUM_PYRAMID_LEVELS)
	{
		int scaledXRes = xRes >> level;
		int scaledYRes = yRes >> level;

		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(scaledXRes)/float(tileSize)), 
			(int)ceil(float(scaledYRes)/float(tileSize)));


		sendFloat3SOAToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo,  rgbMap.x[level], rgbMap.y[level], rgbMap.z[level],  1.0,
			scaledXRes, scaledYRes, xRes, yRes);
	}

}



__host__ void drawCurvaturetoPBO(float4* pbo, float* curvature, int xRes, int yRes)
{
	int tileSize = 16;


	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));


	sendFloat1SOAToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo,  curvature, 0.0,
		xRes, yRes, xRes, yRes);


}


__global__ void sendInt1SOAToPBO(float4* pbo, int* x_src, int xRes, int yRes, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * xRes) + x;
	int pboi = (y * pboXRes) + x;

	if(y < yRes && x < xRes){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = x_src[i];
		pbo[pboi].y = 0.0;
		pbo[pboi].z = 0.0;
		pbo[pboi].w = 0.0;
	}
}



__host__ void drawNormalVoxelsToPBO(float4* pbo, int* voxels, int pboXRes, int pboYRes, int voxelAzimuthBins, int voxelPolarBins)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(voxelAzimuthBins)/float(tileSize)), 
		(int)ceil(float(voxelPolarBins)/float(tileSize)));


	sendInt1SOAToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo,  voxels, voxelAzimuthBins, voxelPolarBins, pboXRes, pboYRes);

}


__global__ void sendHistogramSOAToPBO(float4* pbo, int* binsX, int* binsY, int* binsZ, int length, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pboi = (y * pboXRes) + x;

	if(y < pboYRes && x < length){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = binsX[x];
		pbo[pboi].y = binsY[x];
		pbo[pboi].z = binsZ[x];
		pbo[pboi].w = 0.0;
	}
}




__host__ void drawDecoupledHistogramsToPBO(float4* pbo, Int3SOA histograms,  int length, int pboXRes, int pboYRes)
{

	int tileSize = 16;
	assert(length < pboXRes);

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(length)/float(tileSize)), 
		(int)ceil(float(pboYRes)/float(tileSize)));

	sendHistogramSOAToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo,  histograms.x, histograms.y, histograms.z, length, pboXRes, pboYRes);
}



__global__ void sendInt3SOAToPBO(float4* pbo, int* x_src, int* y_src, int* z_src, int xRes, int yRes, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * xRes) + x;
	int pboi = (y * pboXRes) + x;

	if(y < yRes && x < xRes){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = x_src[i];
		pbo[pboi].y = y_src[i];
		pbo[pboi].z = z_src[i];
		pbo[pboi].w = 0.0;
	}
}




__global__ void sendSegmentDataToPBO(float4* pbo, int* segments, float* projectedDistances, int xRes, int yRes, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = (y * xRes) + x;
	int pboi = (y * pboXRes) + x;

	if(y < yRes && x < xRes){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = segments[i];
		pbo[pboi].y = projectedDistances[i];
		pbo[pboi].z = 0.0;
		pbo[pboi].w = 0.0;
	}
}



__host__ void drawNormalSegmentsToPBO(float4* pbo, int* normalSegments, float* projectedDistanceMap, 
									  int xRes, int yRes, int pboXRes, int pboYRes)
{

	int tileSize = 16;


	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));


	sendSegmentDataToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(pbo,  normalSegments, projectedDistanceMap,
		xRes, yRes, pboXRes, pboYRes);
}




__global__ void drawScaledHistogramToPBOKernel(float4* pbo, int* hist, glm::vec3 color, float scaleInv, int length, int pboXRes, int pboYRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pboi = (y * pboXRes) + x;

	if(y < pboYRes && x < length){

		// Each thread writes one pixel location in the texture (textel)
		pbo[pboi].x = color.x;
		pbo[pboi].y = color.y;
		pbo[pboi].z = color.z;
		pbo[pboi].w = hist[x] * scaleInv;
	}
}



__host__ void drawScaledHistogramToPBO(float4* pbo, int* histogram, glm::vec3 color, int maxValue, int length, int pboXRes, int pboYRes)
{
	int tileSize = 16;
	assert(length < pboXRes);

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(length)/float(tileSize)), 
		(int)ceil(float(pboYRes)/float(tileSize)));


	drawScaledHistogramToPBOKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(pbo, histogram, color, float(1.0f/maxValue), length, pboXRes, pboYRes);
}

