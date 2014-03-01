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
				//Flip towards camera
				norm = -norm;
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

#pragma region  Filtered Average Gradient


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
			//Flip towards camera
			norm = -norm;
		}

		x_norm[i] = norm.x;
		y_norm[i] = norm.y;
		z_norm[i] = norm.z;
		curvature[i] = 0.0f;//filler. Simple normals has no means of estimating curvature


	}
}

__host__ void computeAverageGradientNormals(Float3SOAPyramid horizontalGradient, Float3SOAPyramid vertGradient, 
											Float3SOAPyramid nmap, Float1SOAPyramid curvature, int xRes, int yRes)
{
	int tileSize = 16;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));


	normalsFromGradientKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(horizontalGradient.x[0], horizontalGradient.y[0], horizontalGradient.z[0],
		vertGradient.x[0], vertGradient.y[0], vertGradient.z[0],
		nmap.x[0], nmap.y[0], nmap.z[0], curvature.x[0],
		xRes, yRes);

}

#pragma endregion

#pragma region Eigen Normal Calculation

#define PCA_TILE_SIZE 16
#define PCA_WINDOW_RADIUS 2
#define PCA_MIN_NEIGHBORS 16

__global__ void pcaNormalsKernel(float* vmapX, float* vmapY, float* vmapZ, float* nmapX, float* nmapY, float* nmapZ, float* curvature,
								 int xRes, int yRes, float radiusMeters)
{
	__shared__ float s_positions[3][PCA_TILE_SIZE+2*PCA_WINDOW_RADIUS][PCA_TILE_SIZE+2*PCA_WINDOW_RADIUS];


	//Index of this thread's work target
	int loadBx = blockIdx.x*blockDim.x;
	int loadBy = blockIdx.y*blockDim.y;
	int resultsX = PCA_TILE_SIZE/2 + loadBx  + threadIdx.x;
	int resultsY = PCA_TILE_SIZE/2 + loadBy  + threadIdx.y;
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
				int loadI = (loadBy + loadY)*xRes + (loadBx + loadX);
				//Tiles garunteed to be in range of original image by block layout
				s_positions[0][loadY - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)][loadX - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)] = vmapX[loadI];
				s_positions[1][loadY - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)][loadX - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)] = vmapY[loadI];
				s_positions[2][loadY - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)][loadX - (PCA_TILE_SIZE/2 - PCA_WINDOW_RADIUS)] = vmapZ[loadI];
			}
		}
	}

	__syncthreads();

	nmapX[i] = s_positions[0][threadIdx.y+PCA_WINDOW_RADIUS][threadIdx.x+PCA_WINDOW_RADIUS];
	nmapY[i] = s_positions[1][threadIdx.y+PCA_WINDOW_RADIUS][threadIdx.x+PCA_WINDOW_RADIUS];//
	nmapZ[i] = s_positions[2][threadIdx.y+PCA_WINDOW_RADIUS][threadIdx.x+PCA_WINDOW_RADIUS];//
	curvature[i] = 0.0f;

}


__host__ void computePCANormals(Float3SOAPyramid vmap, Float3SOAPyramid nmap, Float1SOAPyramid curvaturemap, 
								int xRes, int yRes, float radiusMeters)
{

	dim3 threads(PCA_TILE_SIZE, PCA_TILE_SIZE);
	dim3 blocks((int)ceil(float(xRes)/float(PCA_TILE_SIZE))-1, 
		(int)ceil(float(yRes)/float(PCA_TILE_SIZE))-1);


	pcaNormalsKernel<<<blocks,threads>>>(vmap.x[0], vmap.y[0], vmap.z[0], nmap.x[0], nmap.y[0], nmap.z[0], curvaturemap.x[0],
		xRes, yRes, radiusMeters);


}

#pragma endregion