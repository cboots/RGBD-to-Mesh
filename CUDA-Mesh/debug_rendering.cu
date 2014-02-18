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

__global__ void sendPCBToPBOs(float4* dptrPosition, float4* dptrColor, float4* dptrNormal, glm::vec2 resolution, PointCloud* dev_pcb)
{
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * resolution.x) + c;

	if(r<resolution.y && c<resolution.x){

		PointCloud point = dev_pcb[i];

		// Each thread writes one pixel location in the texture (textel)

		dptrPosition[i].x = point.pos.x;
		dptrPosition[i].y = point.pos.y;
		dptrPosition[i].z = point.pos.z;
		dptrPosition[i].w = 1.0f;

		dptrColor[i].x = point.color.r;
		dptrColor[i].y = point.color.g;
		dptrColor[i].z = point.color.b;
		dptrColor[i].w = 1.0f;

		dptrNormal[i].x = point.normal.x;
		dptrNormal[i].y = point.normal.y;
		dptrNormal[i].z = point.normal.z;
		dptrNormal[i].w = 0.0f;
	}
}

// Draws depth image buffer to the texture.
// Texture width and height must match the resolution of the depth image.
// Returns false if width or height does not match, true otherwise
__host__ void drawDepthImageBufferToPBO(float4* dev_PBOpos, DPixel* dev_depthImageBuffer, int texWidth, int texHeight)
{
	int tileSize = 8;

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
	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(texWidth)/float(tileSize)), 
		(int)ceil(float(texHeight)/float(tileSize)));

	sendColorImageBufferToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_PBOpos, glm::vec2(texWidth, texHeight), dev_colorImageBuffer);

}


// Renders various debug information about the 2D point cloud buffer to the texture.
// Texture width and height must match the resolution of the point cloud buffer.
// Returns false if width or height does not match, true otherwise
__host__ void drawPCBToPBO(float4* dptrPosition, float4* dptrColor, float4* dptrNormal, PointCloud* dev_pcb, int texWidth, int texHeight)
{
	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid( (int)ceil(float(texWidth)/float(tileSize)), 
		(int)ceil(float(texHeight)/float(tileSize)) );

	sendPCBToPBOs<<<fullBlocksPerGrid, threadsPerBlock>>>(dptrPosition, dptrColor, dptrNormal, glm::vec2(texWidth, texHeight), dev_pcb);

}
