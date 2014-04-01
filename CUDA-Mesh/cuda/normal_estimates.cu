#include "normal_estimates.h"

#pragma region Simple Normals Calculation



__global__ void simpleNormalsKernel(float* x_vert, float* y_vert, float* z_vert, 
									float* x_norm, float* y_norm, float* z_norm,
									int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){

		glm::vec3 norm = glm::vec3(CUDART_NAN_F);
		float dx1, dy1, dz1, dx2, dy2, dz2;
		if(u < xRes - 1 && v < yRes - 1 && u > 0 && v > 0)
		{

			//Diff to right
			dx1 = x_vert[i+1] - x_vert[i-1];
			dy1 = y_vert[i+1] - y_vert[i-1];
			dz1 = z_vert[i+1] - z_vert[i-1];

			//Diff to bottom
			dx2 = x_vert[i+xRes] - x_vert[i-xRes];
			dy2 = y_vert[i+xRes] - y_vert[i-xRes];
			dz2 = z_vert[i+xRes] - z_vert[i-xRes];



			//d1 cross d2
			norm.x = dy1*dz2-dz1*dy2;
			norm.y = dz1*dx2-dx1*dz2;
			norm.z = dx1*dy2-dy1*dx2;

			if(norm.z < 0.0f)
			{
				//Flip towards camera
				norm = -norm;
			}

			float length = glm::length(norm);
			norm /= length;

		}

		x_norm[i] = norm.x;
		y_norm[i] = norm.y;
		z_norm[i] = norm.z;

	}

}

__host__ void simpleNormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, int numLevels, int xRes, int yRes)
{
	int tileSize = 16;
	
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	simpleNormalsKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(vmap.x[0], vmap.y[0], vmap.z[0],
		nmap.x[0], nmap.y[0], nmap.z[0],
		xRes, yRes);
	
}

#pragma endregion

#pragma region  Filtered Average Gradient


__global__ void normalsFromGradientKernel(float* horizontalGradientX, float* horizontalGradientY, float* horizontalGradientZ,
										  float* vertGradientX, float* vertGradientY, float* vertGradientZ,
										  float* x_norm, float* y_norm, float* z_norm,
										  int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){
		glm::vec3 vertGradient =  glm::vec3(vertGradientX[i], vertGradientY[i], vertGradientZ[i]);
		glm::vec3 horGradient = glm::vec3(horizontalGradientX[i], horizontalGradientY[i], horizontalGradientZ[i]);

		
		glm::vec3 norm = glm::cross(vertGradient, horGradient);

		if(norm.z < 0.0f)
		{
			//Flip towards camera
			norm = -norm;
		}

		float length = glm::length(norm);
		norm /= length;

		

		x_norm[i] = norm.x;
		y_norm[i] = norm.y;
		z_norm[i] = norm.z;


	}
}

__host__ void computeAverageGradientNormals(Float3SOAPyramid horizontalGradient, Float3SOAPyramid vertGradient, 
											Float3SOAPyramid nmap, int xRes, int yRes)
{
	int tileSize = 16;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));


	normalsFromGradientKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(horizontalGradient.x[0], horizontalGradient.y[0], horizontalGradient.z[0],
		vertGradient.x[0], vertGradient.y[0], vertGradient.z[0],
		nmap.x[0], nmap.y[0], nmap.z[0],
		xRes, yRes);
}

#pragma endregion

#pragma region Eigen Normal Calculation

#define PCA_TILE_SIZE 16
#define PCA_WINDOW_RADIUS 2
#define PCA_MIN_NEIGHBORS 16

__device__ float distanceSq(glm::vec3 p1, glm::vec3 p2)
{
	float dx = p1.x-p2.x;
	float dy = p1.y-p2.y;
	float dz = p1.z-p2.z;

	return dx*dx+dy*dy+dz*dz;
}

#define EPSILON 0.0001f
#define PI 3.141592653589f

__device__ glm::vec3 normalFrom3x3Covar(glm::mat3 A, float& curvature) {
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
		eigs[2] = q + 2*p*glm::cos(phi + 2*PI/3);
		eigs[1] = 3*q - eigs[0] - eigs[2];

	}

	//N = (A-eye(3)*eig1)*(A(:,1)-[1;0;0]*eig2);
	glm::mat3 Aeig1 = A;
	Aeig1[0][0] -= eigs[0];
	Aeig1[1][1] -= eigs[0];
	Aeig1[2][2] -= eigs[0];
	normal = Aeig1*(A[0] - glm::vec3(eigs[1],0.0f,0.0f));

	// check if point cloud region is "flat" enough
	curvature = eigs[2]/(eigs[0]+eigs[1]+eigs[2]);


	float length = glm::length(normal);
	normal /= length;
	return normal;
}


__global__ void pcaNormalsKernel(float* vmapX, float* vmapY, float* vmapZ, float* nmapX, float* nmapY, float* nmapZ, float* curvature,
								 int xRes, int yRes, float radiusMetersSq)
{
	__shared__ float s_positions[3][PCA_TILE_SIZE+2*PCA_WINDOW_RADIUS][PCA_TILE_SIZE+2*PCA_WINDOW_RADIUS];


	//Upper left corner of aligned loading block
	int loadBx = blockIdx.x*blockDim.x-PCA_TILE_SIZE/2;
	int loadBy = blockIdx.y*blockDim.y-PCA_TILE_SIZE/2;

	//Index of this thread's work target
	int resultsX = blockIdx.x*blockDim.x + threadIdx.x;
	int resultsY = blockIdx.y*blockDim.y + threadIdx.y;
	int i = resultsX + xRes*resultsY;

	int loadX = threadIdx.x + 16 * (threadIdx.y % 2);
	//Offset to shared memory is threadIdx.x-PCA_WINDOW_RADIUS, threadIdx.y-PCA_WINDOW_RADIUS
	if(loadX >= PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS && loadX < PCA_TILE_SIZE*3/2 + PCA_WINDOW_RADIUS){
		//Is in horizontal range. Only need to perform vertical check in loop

#pragma unroll
		for(int istep = 0; istep < 3; ++istep)
		{
			int loadY = istep*PCA_TILE_SIZE/2 + threadIdx.y/2;
			if(loadY >= PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS && loadY < PCA_TILE_SIZE*3/2 + PCA_WINDOW_RADIUS)
			{
				bool loadIdInRange = (loadBy + loadY) > 0 && (loadBy + loadY) < yRes && (loadBx + loadX) > 0 && (loadBx + loadX) < xRes;

				int loadI = (loadBy + loadY)*xRes + (loadBx + loadX);

				//Tiles garunteed to be in range of original image by block layout
				s_positions[0][loadY - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)][loadX - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)] = loadIdInRange?vmapX[loadI]:CUDART_NAN_F;
				s_positions[1][loadY - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)][loadX - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)] = loadIdInRange?vmapY[loadI]:CUDART_NAN_F;
				s_positions[2][loadY - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)][loadX - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)] = loadIdInRange?vmapZ[loadI]:CUDART_NAN_F;
			}
		}
	}

	__syncthreads();

	//Load done
	//Compute centroid. 
	//Use center point to determine point range
	int sx = threadIdx.x+PCA_WINDOW_RADIUS;
	int sy = threadIdx.y+PCA_WINDOW_RADIUS;
	glm::vec3 centerPos = glm::vec3(s_positions[0][sy][sx],
		s_positions[1][sy][sx],
		s_positions[2][sy][sx]);

	int neighborCount = 0;
	glm::vec3 centroid = glm::vec3();
#pragma unroll
	for(int x = -PCA_WINDOW_RADIUS; x <= PCA_WINDOW_RADIUS; ++x)
	{
#pragma unroll
		for(int y = -PCA_WINDOW_RADIUS; y <= PCA_WINDOW_RADIUS; ++y)
		{
			glm::vec3 p = glm::vec3(s_positions[0][sy+y][sx+x],
				s_positions[1][sy+y][sx+x],
				s_positions[2][sy+y][sx+x]);
			glm::vec3 diff = p-centerPos;
			if(glm::dot(diff,diff) <= radiusMetersSq)
			{
				//In range
				neighborCount++;
				centroid += p;
			}
		}
	}
	centroid /= neighborCount;


	neighborCount = 0;
	glm::mat3 covariance = glm::mat3(0.0f);
	//At this point, we have a true centroid
#pragma unroll
	for(int x = -PCA_WINDOW_RADIUS; x <= PCA_WINDOW_RADIUS; ++x)
	{
#pragma unroll
		for(int y = -PCA_WINDOW_RADIUS; y <= PCA_WINDOW_RADIUS; ++y)
		{
			glm::vec3 p = glm::vec3(s_positions[0][sy+y][sx+x],
				s_positions[1][sy+y][sx+x],
				s_positions[2][sy+y][sx+x]);
			glm::vec3 diff = p-centroid;
			if(glm::dot(diff,diff) <= radiusMetersSq)
			{
				//In range
				neighborCount++;
				covariance[0] += diff.x*diff;
				covariance[1] += diff.y*diff;
				covariance[2] += diff.z*diff;
			}
		}
	}
	covariance /= neighborCount - 1;
	//Compute eigenvalue
	float curve = CUDART_NAN_F;

	glm::vec3 norm = glm::vec3(CUDART_NAN_F);
	if(neighborCount >= PCA_MIN_NEIGHBORS){
		norm = normalFrom3x3Covar(covariance, curve);
	}
	nmapX[i] = norm.x;
	nmapY[i] = norm.y;//
	nmapZ[i] = norm.z;//
	curvature[i] = curve;

}


__host__ void computePCANormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, float* curvature, 
								int xRes, int yRes, float radiusMeters)
{
	assert(PCA_WINDOW_RADIUS < PCA_TILE_SIZE / 2);

	dim3 threads(PCA_TILE_SIZE, PCA_TILE_SIZE);
	dim3 blocks((int)ceil(float(xRes)/float(PCA_TILE_SIZE)), 
		(int)ceil(float(yRes)/float(PCA_TILE_SIZE)));


	pcaNormalsKernel<<<blocks,threads>>>(vmap.x[0], vmap.y[0], vmap.z[0], nmap.x[0], nmap.y[0], nmap.z[0], curvature,
		xRes, yRes, radiusMeters*radiusMeters);


}

#pragma endregion


#pragma region Normal Cartesian-Spherical conversions
//Assumes normalized vectors
__global__ void normalsToSpherical(float* normX, float* normY, float* normZ, float* azimuthAngle, float* polarAngle, int arraySize)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if(i < arraySize)
	{
		float x = normX[i];
		float y = normY[i];
		float z = normZ[i];

		polarAngle[i] = acosf(-z);
		float azimuth = atan2f(y,x);
		azimuthAngle[i] = (azimuth < 0.0f)?(azimuth + 2.0*PI):azimuth;//Unwrap azimuth angle
	}
}

__host__ void convertNormalToSpherical(float* normX, float* normY, float* normZ, float* azimuthAngle, float* polarAngle, int arraySize)
{
	int BLOCK_SIZE = 256;

	dim3 blocks((int)ceil(float(arraySize)/float(BLOCK_SIZE)));
	dim3 threads(BLOCK_SIZE);

	normalsToSpherical<<<blocks,threads>>>(normX, normY, normZ, azimuthAngle, polarAngle, arraySize);
}

#pragma endregion


#pragma region Curvature

#define PI_INV_F 0.3183098862f

__global__ void estimateCurvatureKernel(float* x_norm, float* y_norm, float* z_norm,
										float* curvature,
										int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){
		//Curvature estimate
		float curve = CUDART_NAN_F;
		if(u < xRes - 1 && v < yRes - 1)
		{
			glm::vec3 normThis = glm::vec3(x_norm[i], y_norm[i], z_norm[i]);
			glm::vec3 normRight = glm::vec3(x_norm[i+1], y_norm[i+1], z_norm[i+1]);
			glm::vec3 normBelow = glm::vec3(x_norm[i+xRes], y_norm[i+xRes], z_norm[i+xRes]);

			float dotProd = glm::max(glm::dot(normRight, normThis),glm::dot(normBelow, normThis));
			//curve = acosf(dotProd)/sqrtf(dx*dx+dy*dy+dz*dz);
			curve = acosf(dotProd)*PI_INV_F;


		}

		curvature[i] = curve;
	}
}


__host__ void curvatureEstimate(Float3SOAPyramid nmap, float* curvature, int xRes, int yRes)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));
	estimateCurvatureKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(	nmap.x[0], nmap.y[0], nmap.z[0], curvature,	xRes, yRes);
}

#pragma endregion