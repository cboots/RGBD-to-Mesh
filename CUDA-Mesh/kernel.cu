#include "Device.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <cub/cub.cuh>
#define CUB_CDP

ColorPixel* dev_colorImageBuffer;
DPixel* dev_depthImageBuffer;
PointCloud* dev_pointCloudBuffer;
void* dev_compactionTempStorage;
size_t dev_compactionTempStorageBytes;
int* dev_compactionNumValid;
IsValidNormal selectOp;

int	cuImageWidth = 0;
int	cuImageHeight = 0;

GLuint imagePBO = (GLuint)NULL;

__host__ void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		exit(EXIT_FAILURE); 
	}
}

__global__ void makePointCloud(ColorPixel* colorPixels, DPixel* dPixels, int xRes, int yRes, PointCloud* pointCloud) {
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * xRes) + c;

	if(r < yRes && c < xRes) {
		// In range
		if (dPixels[i].depth != 0) {
			float u = (c - (xRes-1)/2.0f + 1) / (xRes-1); // image plane u coordinate
			float v = ((yRes-1)/2.0f - r) / (yRes-1); // image plane v coordinate
			float Z = dPixels[i].depth/1000.0f; // depth converted to meters
			pointCloud[i].pos = glm::vec3(u*Z*SCALE_X, v*Z*SCALE_Y, -Z); // convert uv to XYZ
			pointCloud[i].color = glm::vec3(colorPixels[i].r/255.0f, colorPixels[i].g/255.0f, colorPixels[i].b/255.0f); // copy over texture
		} else {
			pointCloud[i].pos = glm::vec3(0.0f);
			pointCloud[i].color = glm::vec3(0.0f);
		}
		// Always clear normals
		pointCloud[i].normal = glm::vec3(0.0f);
	}
}

__device__ glm::vec3 normalFrom3x3Covar(glm::mat3 A) {
	// Given a (real, symmetric) 3x3 covariance matrix A, returns the eigenvector corresponding to the min eigenvalue
	// (see: http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices)
	glm::vec3 eigs;
	glm::vec3 normal = glm::vec3(0.0f);
	float p1 = pow(A[0][1], 2) + pow(A[0][2], 2) + pow(A[1][2], 2);
	if (abs(p1) < EPSILON) { // A is diagonal
		eigs = glm::vec3(A[0][0], A[1][1], A[2][2]);
	} else {
		float q = (A[0][0] + A[1][1] + A[2][2])/3.0f; // mean(trace(A))
		float p2 = pow(A[0][0]-q, 2) + pow(A[1][1]-q, 2) + pow(A[2][2]-q, 2) + 2*p1;
		float p = sqrt(p2/6);
		glm::mat3 B = (1/p) * (A-q*glm::mat3(1.0f));
		float r = glm::determinant(B)/2;
		// theoretically -1 <= r <= 1, but clamp in case of numeric error
		float phi;
		if (r <= -1) {
			phi = PI / 3;
		} else if (r >= 1) { 
			phi = 0;
		} else {
			phi = glm::acos(r)/3;
		}
		eigs[0] = q + 2*p*glm::cos(phi);
		eigs[1] = q + 2*p*glm::cos(phi + 2*PI/3);
		eigs[2] = 3*q - eigs.x - eigs.z;
		float tmp;
		int i, eig_i;
		// sorting: swap first pair if necessary, then second pair, then first pair again
		for (i=0; i<3; i++) {
			eig_i = i%2;
			tmp = eigs[eig_i];
			eigs[eig_i] = glm::min(tmp, eigs[eig_i+1]);
			eigs[eig_i+1] = glm::max(tmp, eigs[eig_i+1]);
		}
	}
	// check if point cloud region is "flat" enough
	if (eigs[1]/eigs[0] >= MIN_EIG_RATIO) {
		normal = glm::cross(A[0] - glm::vec3(eigs[0], 0.0f, 0.0f), A[1] - glm::vec3(0.0f, eigs[0], 0.0f));
	}
	return glm::normalize(normal);
}

__global__ void computePointNormals(PointCloud* pointCloud, int xRes, int yRes) {
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
    int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (r * xRes) + c;

    int N = 0; // number of nearest neighbors

    glm::vec3 neighbor;
    glm::vec3 center = pointCloud[i].pos;
    glm::mat3 covariance = glm::mat3(0.0f);
	int win_r, win_c, win_i;
	for (win_r = r-RAD_WIN; win_r <= r+RAD_WIN; win_r++) {
		for (win_c = c-RAD_WIN; win_c <= c+RAD_WIN; win_c++) {
			// exclude center from neighbor search
			if (win_r != r && win_c != c) {
                // check if neighbor is in frame
                if (win_r >= 0 && win_r < yRes && win_c >= 0 && win_c < xRes) {
                    win_i = (win_r * xRes) + win_c;
                    neighbor = pointCloud[win_i].pos;
                    // check if neighbor has valid depth data
                    if (glm::length(neighbor) > EPSILON) {
                        // check if neighbor is close enough in world space
                        if (glm::distance(neighbor, center) < RAD_NN) {
                            N += 1; // valid neighbor found
                            glm::vec3 difference = neighbor - center;
                            // remember GLM is column major
                            covariance[0] += (difference * difference[0]);
                            covariance[1] += (difference * difference[1]);
                            covariance[2] += (difference * difference[2]);
                        }
                    }
                }
			}
		}
	}
    // check if enough nearest neighbors were found
    if (N >= MIN_NN) {
        covariance = covariance/N; // average covariance
        // compute and assign normal (0 if not "flat" enough)
        pointCloud[i].normal = normalFrom3x3Covar(covariance);
    }
}

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

__host__ void deletePBO(GLuint *pbo)
{
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

// Intialize pipeline buffers
__host__ void initCuda(int width, int height)
{
	// Allocate buffers
	cudaMalloc((void**) &dev_colorImageBuffer, sizeof(ColorPixel)*width*height);
	cudaMalloc((void**) &dev_depthImageBuffer, sizeof(DPixel)*width*height);
	cudaMalloc((void**) &dev_pointCloudBuffer, sizeof(PointCloud)*width*height);
	cuImageWidth = width;
	cuImageHeight = height;

    // Set up CUB DeviceSelectIf call for normals compaction
    // see: http://nvlabs.github.io/cub/structcub_1_1_device_select.html
    void* dev_compactionTempStorage = NULL;
    dev_compactionTempStorageBytes = 0;
    cudaMalloc(&dev_compactionNumValid, sizeof(int));
    cub::DeviceSelect::If(dev_compactionTempStorage, dev_compactionTempStorageBytes, dev_pointCloudBuffer, dev_pointCloudBuffer, dev_compactionNumValid, width*height, selectOp); 
    cudaMalloc(&dev_compactionTempStorage, dev_compactionTempStorageBytes);
}

// Free all allocated buffers and close out environment
__host__ void cleanupCuda()
{
	if(imagePBO) deletePBO(&imagePBO);

	cudaFree(dev_colorImageBuffer);
	cudaFree(dev_depthImageBuffer);
	cudaFree(dev_pointCloudBuffer);
    cudaFree(dev_compactionTempStorage);
    cudaFree(dev_compactionNumValid);
	cuImageWidth = 0;
	cuImageHeight = 0;

	cudaDeviceReset();

}

// Copies a depth image to the GPU buffer. 
// Returns false if width and height do not match buffer size set by initCuda(), true if success
__host__ bool pushDepthArrayToBuffer(DPixel* hDepthArray, int width, int height)
{
	if(width != cuImageWidth || height != cuImageHeight)
		return false;//Buffer wrong size

	cudaMemcpy(dev_depthImageBuffer, hDepthArray, sizeof(DPixel)*width*height, cudaMemcpyHostToDevice);
	return true;
}

// Copies a color image to the GPU buffer. 
// Returns false if width and height do not match buffer size set by initCuda(), true if success
__host__ bool pushColorArrayToBuffer(ColorPixel* hColorArray, int width, int height)
{
	if(width != cuImageWidth || height != cuImageHeight)
		return false; //Buffer wrong size

	cudaMemcpy((void*)dev_colorImageBuffer, hColorArray, sizeof(ColorPixel)*width*height, cudaMemcpyHostToDevice);
	return true;
}

// Converts the color and depth images currently in GPU buffers into point cloud buffer
__host__ void convertToPointCloud()
{
	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(cuImageWidth)/float(tileSize)), 
		(int)ceil(float(cuImageHeight)/float(tileSize)));

	makePointCloud<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_colorImageBuffer, dev_depthImageBuffer, cuImageWidth, cuImageHeight, dev_pointCloudBuffer);
}

// Computes normals for point cloud in buffer and writes back to the point cloud buffer.
__host__ void computePointCloudNormals()
{
	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(cuImageWidth)/float(tileSize)), 
		(int)ceil(float(cuImageHeight)/float(tileSize)));

	computePointNormals<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_pointCloudBuffer, cuImageWidth, cuImageHeight);
}


// Draws depth image buffer to the texture.
// Texture width and height must match the resolution of the depth image.
// Returns false if width or height does not match, true otherwise
bool drawDepthImageBufferToPBO(float4* dev_PBOpos, int texWidth, int texHeight)
{
	if(texWidth != cuImageWidth || texHeight != cuImageHeight)
		return false;

	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(texWidth)/float(tileSize)), 
		(int)ceil(float(texHeight)/float(tileSize)));

	sendDepthImageBufferToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_PBOpos, glm::vec2(texWidth, texHeight), dev_depthImageBuffer);

	return true;
}

// Draws color image buffer to the texture.
// Texture width and height must match the resolution of the color image.
// Returns false if width or height does not match, true otherwise
// dev_PBOpos must be a CUDA device pointer
bool drawColorImageBufferToPBO(float4* dev_PBOpos, int texWidth, int texHeight)
{
	if(texWidth != cuImageWidth || texHeight != cuImageHeight)
		return false;

	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(texWidth)/float(tileSize)), 
		(int)ceil(float(texHeight)/float(tileSize)));

	sendColorImageBufferToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_PBOpos, glm::vec2(texWidth, texHeight), dev_colorImageBuffer);

	return true;
}

// Renders the point cloud as stored in the VBO to the texture
__host__ void drawPointCloudVBOToTexture(GLuint texture, int texWidth, int texHeight /*TODO: More vizualization parameters here*/)
{
	// TODO: Implement

}

// Renders various debug information about the 2D point cloud buffer to the texture.
// Texture width and height must match the resolution of the point cloud buffer.
// Returns false if width or height does not match, true otherwise
__host__ bool drawPCBToPBO(float4* dptrPosition, float4* dptrColor, float4* dptrNormal, int texWidth, int texHeight)
{
	if(texWidth != cuImageWidth || texHeight != cuImageHeight)
		return false;

	int tileSize = 8;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid( (int)ceil(float(texWidth)/float(tileSize)), 
		                    (int)ceil(float(texHeight)/float(tileSize)) );

	sendPCBToPBOs<<<fullBlocksPerGrid, threadsPerBlock>>>(dptrPosition, dptrColor, dptrNormal, glm::vec2(texWidth, texHeight), dev_pointCloudBuffer);

	return true;
}

// Takes a device pointer to the point cloud VBO and copies the contents of the PointCloud buffer to the VBO using stream compaction.
// See: http://nvlabs.github.io/cub/structcub_1_1_device_select.html
__host__ int compactPointCloudToVBO(PointCloud* vbo) {
    int numValid[1];
    cub::DeviceSelect::If(dev_compactionTempStorage, dev_compactionTempStorageBytes, dev_pointCloudBuffer, vbo, dev_compactionNumValid, cuImageWidth*cuImageHeight, selectOp);
    cudaMemcpy(numValid, dev_compactionNumValid, sizeof(int), cudaMemcpyDeviceToHost);
    return numValid[0];
}