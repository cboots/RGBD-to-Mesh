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

			
			//if n dot p > 0, flip towards viewpoint
			if(norm.x*x_vert[i] + norm.y*y_vert[i] + norm.z * z_vert[i] > 0.0f)
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
										  float* x_vert, float* y_vert, float* z_vert, 
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
		
		//if n dot p > 0, flip towards viewpoint
		if(norm.x*x_vert[i] + norm.y*y_vert[i] + norm.z * z_vert[i] > 0.0f)
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
											Float3SOAPyramid vmap, Float3SOAPyramid nmap, int xRes, int yRes)
{
	int tileSize = 16;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));


	normalsFromGradientKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(horizontalGradient.x[0], horizontalGradient.y[0], horizontalGradient.z[0],
		vertGradient.x[0], vertGradient.y[0], vertGradient.z[0],
		vmap.x[0], vmap.y[0], vmap.z[0],
		nmap.x[0], nmap.y[0], nmap.z[0],
		xRes, yRes);
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
		azimuthAngle[i] = (azimuth < 0.0f)?(azimuth + 2.0*PI_F):azimuth;//Unwrap azimuth angle
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
