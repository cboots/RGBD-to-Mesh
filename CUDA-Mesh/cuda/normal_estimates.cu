#include "normal_estimates.h"

#pragma region Simple Normals Calculation

__global__ void simpleNormalsKernel(float* x_vert, float* y_vert, float* z_vert, 
									float* x_norm, float* y_norm, float* z_norm,
									float* curvature,
									int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){

		glm::vec3 norm = glm::vec3(CUDART_NAN_F);

		if(u < xRes - 1 && v < yRes - 1 && u > 0 && v > 0)
		{

			//Diff to right
			float dx1 = x_vert[i+1] - x_vert[i-1];
			float dy1 = y_vert[i+1] - y_vert[i-1];
			float dz1 = z_vert[i+1] - z_vert[i-1];

			//Diff to bottom
			float dx2 = x_vert[i+xRes] - x_vert[i-xRes];
			float dy2 = y_vert[i+xRes] - y_vert[i-xRes];
			float dz2 = z_vert[i+xRes] - z_vert[i-xRes];

			//d1 cross d2
			norm.x = dy1*dz2-dz1*dy2;
			norm.y = dz1*dx2-dx1*dz2;
			norm.z = dx1*dy2-dy1*dx2;

			if(norm.z > 0.0f)
			{
				norm.z = -norm.z;
			}

			norm = glm::normalize(norm);

		}

		x_norm[i] = norm.x;
		y_norm[i] = norm.y;
		z_norm[i] = norm.z;
		curvature[i] = 0.0f;//filler. Simple normals has no means of estimating curvature


	}

}

__host__ void simpleNormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, Float1SOAPyramid curvaturemap, int numLevels, int xRes, int yRes)
{
	int tileSize = 16;

	for(int i = 0; i < numLevels; ++i)
	{
		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(xRes>>i)/float(tileSize)), 
			(int)ceil(float(yRes>>i)/float(tileSize)));


		simpleNormalsKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(vmap.x[i], vmap.y[i], vmap.z[i],
			nmap.x[i], nmap.y[i], nmap.z[i], curvaturemap.x[i],
			xRes>>i, yRes>>i);
	}
}

#pragma endregion

#pragma region Integral Image Average Gradient

__global__ void normalsFromGradientKernel(float* horizontalGradientX, float* horizontalGradientY, float* horizontalGradientZ,
										  float* vertGradientX, float* vertGradientY, float* vertGradientZ,
										  float* x_norm, float* y_norm, float* z_norm, float* curvature,
										  int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){

		glm::vec3 norm = glm::normalize(glm::cross(glm::vec3(vertGradientX[i], vertGradientY[i], vertGradientZ[i]), 
			glm::vec3(horizontalGradientX[i], horizontalGradientY[i], horizontalGradientZ[i])));

		if(norm.z > 0.0f)
		{
			norm.z = -norm.z;
		}

		x_norm[i] = norm.x;
		y_norm[i] = norm.y;
		z_norm[i] = norm.z;
		curvature[i] = 0.0f;//filler. Simple normals has no means of estimating curvature


	}
}

__host__ void computeAverageGradientNormals(Float3SOAPyramid horizontalGradient, Float3SOAPyramid vertGradient, 
											Float3SOAPyramid nmap, Float1SOAPyramid curvature, int level, int xRes, int yRes)
{
	int tileSize = 16;
	int i = level;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes>>i)/float(tileSize)), 
		(int)ceil(float(yRes>>i)/float(tileSize)));


	normalsFromGradientKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(horizontalGradient.x[i], horizontalGradient.y[i], horizontalGradient.z[i],
		vertGradient.x[i], vertGradient.y[i], vertGradient.z[i],
		nmap.x[i], nmap.y[i], nmap.z[i], curvature.x[i],
		xRes>>i, yRes>>i);

}

#pragma endregion

#pragma region Filtered Average Gradient



#pragma endregion

#pragma region Eigen Normal Calculation


#pragma endregion