#include "normal_estimates.h"

__global__ void simpleNormalsKernel(float* x_vert, float* y_vert, float* z_vert, 
									float* x_norm, float* y_norm, float* z_norm,
									int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	int i = v * xRes + u;

	if(u < xRes && v < yRes){

		glm::vec3 norm = glm::vec3(CUDART_NAN_F);

		if(u < xRes - 1 && v < yRes - 1)
		{
			float xc = x_vert[i];
			float yc = y_vert[i];
			float zc = z_vert[i];

			//Diff to right
			float dx1 = x_vert[i+1] - xc;
			float dy1 = y_vert[i+1] - yc;
			float dz1 = z_vert[i+1] - zc;

			//Diff to bottom
			float dx2 = x_vert[i+xRes] - xc;
			float dy2 = y_vert[i+xRes] - yc;
			float dz2 = z_vert[i+xRes] - zc;

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


	}

}

__host__ void simpleNormals(VMapSOA vmap, NMapSOA nmap, int numLevels, int xRes, int yRes)
{
	int tileSize = 16;

	for(int i = 0; i < numLevels - 1; ++i)
	{
		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(xRes>>i)/float(tileSize)), 
			(int)ceil(float(yRes>>i)/float(tileSize)));


		simpleNormalsKernel<<<fullBlocksPerGrid,threadsPerBlock>>>(vmap.x[i], vmap.y[i], vmap.z[i],
			nmap.x[i], nmap.y[i], nmap.z[i],
			xRes>>i, yRes>>i);
	}

}