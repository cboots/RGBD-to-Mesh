#include "preprocessing.h"

__global__ void flipDepthImageXAxisKernel(rgbd::framework::DPixel* dev_depthBuffer, int xRes, int yRes)
{
	extern __shared__ int s_swapbuffer[];

	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int loadIndexX = (blockDim.x/2 * (gridDim.x * 2 - 2 - blockIdx.x)) + threadIdx.x;
	if(threadIdx.x < blockDim.x/2)
		loadIndexX = blockDim.x/2 * blockIdx.x + threadIdx.x;
	

	s_swapbuffer[threadIdx.x + threadIdx.y * blockDim.x] = dev_depthBuffer[loadIndexX + yi*xRes].depth;


	dev_depthBuffer[loadIndexX + yi*xRes].depth = s_swapbuffer[(blockDim.x - 1 - threadIdx.x) + threadIdx.y*blockDim.x];

}

__host__ void flipDepthImageXAxis(rgbd::framework::DPixel* dev_depthBuffer, int xRes, int yRes)
{
	int blockWidth = 32;
	int blockHeight = 8;

	assert(xRes/2 % blockWidth == 0);

	dim3 threadsPerBlock(blockWidth, blockHeight);

	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(blockWidth)), 
		(int)ceil(float(yRes)/float(blockHeight)));
	int sharedCount = blockWidth*blockHeight*sizeof(int);

	flipDepthImageXAxisKernel<<<fullBlocksPerGrid, threadsPerBlock, sharedCount>>>(dev_depthBuffer, xRes, yRes);
}

#pragma region VMap No Filter
__global__ void buildVMapNoFilterKernel(rgbd::framework::DPixel* dev_depthBuffer, Float3SOAPyramid vmapSOA, int xRes, int yRes,
										rgbd::framework::Intrinsics intr, float maxDepth) 
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(u < xRes && v < yRes) 
	{
		int i = (v * xRes) + u;
		int i2 = (v * xRes) + (xRes - 1 - u);//X Flipped index
		float x = CUDART_NAN_F;
		float y = CUDART_NAN_F;
		float z = dev_depthBuffer[i2].depth * 0.001f;

		if (z > 0.001f && z < maxDepth) {//Exclude zero or negative depths.  

			x = (u - intr.cx) * z / intr.fx;
			y = (v - intr.cy) * z / intr.fy;
		} else{
			z = CUDART_NAN_F;
		}

		//Write to SOA in memory coallesed way
		vmapSOA.x[0][i] = x;
		vmapSOA.y[0][i] = y;
		vmapSOA.z[0][i] = z;

	}
}

__host__ void buildVMapNoFilterCUDA(rgbd::framework::DPixel* dev_depthBuffer, Float3SOAPyramid vmapSOA, int xRes, int yRes, 
									rgbd::framework::Intrinsics intr, float maxDepth)
{

	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	buildVMapNoFilterKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_depthBuffer, vmapSOA, xRes, yRes, intr, maxDepth);

}

#pragma endregion


__constant__ float cGaussianSpatialKernel[GAUSSIAN_SPATIAL_KERNEL_SIZE];

__host__ void setGaussianSpatialKernel(float sigma)
{
	float kernel[GAUSSIAN_SPATIAL_KERNEL_SIZE];

	for(int i = -GAUSSIAN_SPATIAL_FILTER_RADIUS; i <= GAUSSIAN_SPATIAL_FILTER_RADIUS; ++i)
	{
		kernel[i+GAUSSIAN_SPATIAL_FILTER_RADIUS] = expf(-i*i/(2*sigma));
	}

	cudaMemcpyToSymbol(cGaussianSpatialKernel, kernel, GAUSSIAN_SPATIAL_KERNEL_SIZE*sizeof(float));
}

#pragma region VMap Seperable Gaussian Kernel
//========Simple Gaussian Kernel Rows=========

//Block width and height. 16 is convenient for memory alignment (4*16==64Bytes)
#define		ROWS_BLOCKDIM_X		16
#define		ROWS_BLOCKDIM_Y		12

//Number of pixels processed per thread
#define		ROWS_RESULT_STEPS	4
//Number of blocks in region. ROWS_HALO_STEPS*ROWS_BLOCKDIM_X must be > KERNEL_RADIUS
#define		ROWS_HALO_STEPS		1


__global__ void gaussianKernelRows(rgbd::framework::DPixel* dev_depthBuffer, Float3SOAPyramid vmapSOA, int xRes, int yRes, float maxDepth)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X*(ROWS_RESULT_STEPS+2*ROWS_HALO_STEPS)];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;


	//Align source pointer
	dev_depthBuffer += baseY*xRes + baseX;//Align source pointer with this thread for convenience

	//Align output pointer
	float* d_Dest = vmapSOA.z[0] + baseY*xRes + baseX;


	//Main data
#pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; ++i)
	{
		float depth = dev_depthBuffer[i * ROWS_BLOCKDIM_X].depth*0.001f;
		if(depth > maxDepth)
			depth = 0.0f;

		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = depth;
	}

	//Left halo
	for(int i = 0; i < ROWS_HALO_STEPS; ++i)
	{
		//Bounds check on source array. Fill with zeros if out of bounds.
		float depth = (baseX >= -i*ROWS_BLOCKDIM_X)? dev_depthBuffer[i * ROWS_BLOCKDIM_X].depth*0.001f : 0.0f;

		//Max depth test
		if(depth > maxDepth)
			depth = 0.0f;

		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = depth;

	}

	//Right halo
	for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; ++i)
	{
		//Bounds check on source array. Fill with zeros if out of bounds.
		float depth = (xRes - baseX > i*ROWS_BLOCKDIM_X)? dev_depthBuffer[i * ROWS_BLOCKDIM_X].depth*0.001f : 0.0f;

		//Max depth test
		if(depth > maxDepth)
			depth = 0.0f;

		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = depth;
	}
	__syncthreads();


	//=======END OF LOADING STAGE======

	//=======BEGIN COMPURE STAGE=======
#pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){//For each result block
		float sum = 0;
		float weightAccum = 0;

#pragma unroll
		for(int j = -GAUSSIAN_SPATIAL_FILTER_RADIUS; j <= GAUSSIAN_SPATIAL_FILTER_RADIUS; j++)
		{
			float depth = s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			if(depth > 0.001f)
			{
				sum +=	cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j] * depth;
				weightAccum += cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j];
			}
		}
		d_Dest[i * ROWS_BLOCKDIM_X] = sum/weightAccum;//Normalize
	}
}


//=========SIMPLE GAUSSIAN KERNEL COLUMNS========
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1

__global__ void gaussianKernelCols(Float3SOAPyramid vmapSOA, int xRes, int yRes, rgbd::framework::Intrinsics intr)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	float* d_x = vmapSOA.x[0] + baseY * xRes + baseX;
	float* d_y = vmapSOA.y[0] + baseY * xRes + baseX;
	float* d_z = vmapSOA.z[0] + baseY * xRes + baseX;

	//Main data
#pragma unroll
	for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_z[i * COLUMNS_BLOCKDIM_Y * xRes];

	//Upper halo
	for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 
		(baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_z[i * COLUMNS_BLOCKDIM_Y * xRes] : 0;

	//Lower halo
	for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
		(yRes - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_z[i * COLUMNS_BLOCKDIM_Y * xRes] : 0;

	//End load step

	__syncthreads();

	//Compute and store results
#pragma unroll
	for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
		float sum = 0;
		float weightAccum = 0;

#pragma unroll
		for(int j = -GAUSSIAN_SPATIAL_FILTER_RADIUS; j <= GAUSSIAN_SPATIAL_FILTER_RADIUS; j++)
		{
			float depth = s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			if(depth > 0.001f){
				sum += cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j] * depth;
				weightAccum += cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j];
			}
		}

		float x = CUDART_NAN_F;
		float y = CUDART_NAN_F;
		float z = sum/weightAccum;//Normalize
		if (z > 0.001f) {//Exclude zero or negative depths.  
			int u = threadIdx.x+blockIdx.x*COLUMNS_BLOCKDIM_X;//Pixel space x coord
			int v = threadIdx.y+(blockIdx.y*COLUMNS_RESULT_STEPS + (i-COLUMNS_HALO_STEPS))*COLUMNS_BLOCKDIM_Y;//pixel space y coord
			x = (u - intr.cx) * z / intr.fx;
			y = (v - intr.cy) * z / intr.fy;
		} else{
			z = CUDART_NAN_F;
		}
		d_x[i * COLUMNS_BLOCKDIM_Y * xRes] = x;
		d_y[i * COLUMNS_BLOCKDIM_Y * xRes] = y;
		d_z[i * COLUMNS_BLOCKDIM_Y * xRes] = z;
	}


}




__host__ void buildVMapGaussianFilterCUDA(rgbd::framework::DPixel* dev_depthBuffer, Float3SOAPyramid vmapSOA, int xRes, int yRes, 
										  rgbd::framework::Intrinsics intr, float maxDepth)
{
	//=====Row filter and maxdepth thresholding step======
	//Assert that kernel parameters are properly memory aligned. Otherwise, kernel will either fail or be inefficient
	assert(GAUSSIAN_SPATIAL_FILTER_RADIUS <= ROWS_BLOCKDIM_X*ROWS_HALO_STEPS);
	assert( xRes % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
	assert( yRes % ROWS_BLOCKDIM_Y == 0 );

	dim3 blocks(xRes / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), yRes / ROWS_BLOCKDIM_Y);
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	gaussianKernelRows<<<blocks, threads>>>(dev_depthBuffer, vmapSOA, xRes, yRes, maxDepth);

	//======Column filter and point cloud generation======
	assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= GAUSSIAN_SPATIAL_FILTER_RADIUS );
    assert( xRes % COLUMNS_BLOCKDIM_X == 0 );
    assert( yRes % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    blocks = dim3(xRes / COLUMNS_BLOCKDIM_X, yRes / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    threads = dim3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	gaussianKernelCols<<<blocks, threads>>>(vmapSOA, xRes, yRes, intr);

}

#pragma endregion

#pragma region VMap Seperable Bilateral Filter
__global__ void bilateralFilterKernelRows(rgbd::framework::DPixel* dev_depthBuffer, Float3SOAPyramid vmapSOA, int xRes, int yRes, float maxDepth, float inv2sig_t)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X*(ROWS_RESULT_STEPS+2*ROWS_HALO_STEPS)];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;


	//Align source pointer
	dev_depthBuffer += baseY*xRes + baseX;//Align source pointer with this thread for convenience

	//Align output pointer
	float* d_Dest = vmapSOA.z[0] + baseY*xRes + baseX;


	//Main data
#pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; ++i)
	{
		float depth = dev_depthBuffer[i * ROWS_BLOCKDIM_X].depth*0.001f;
		if(depth > maxDepth)
			depth = 0.0f;

		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = depth;
	}

	//Left halo
	for(int i = 0; i < ROWS_HALO_STEPS; ++i)
	{
		//Bounds check on source array. Fill with zeros if out of bounds.
		float depth = (baseX >= -i*ROWS_BLOCKDIM_X)? dev_depthBuffer[i * ROWS_BLOCKDIM_X].depth*0.001f : 0.0f;

		//Max depth test
		if(depth > maxDepth)
			depth = 0.0f;

		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = depth;

	}

	//Right halo
	for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; ++i)
	{
		//Bounds check on source array. Fill with zeros if out of bounds.
		float depth = (xRes - baseX > i*ROWS_BLOCKDIM_X)? dev_depthBuffer[i * ROWS_BLOCKDIM_X].depth*0.001f : 0.0f;

		//Max depth test
		if(depth > maxDepth)
			depth = 0.0f;

		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = depth;
	}
	__syncthreads();


	//=======END OF LOADING STAGE======

	//=======BEGIN COMPURE STAGE=======
#pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){//For each result block
		float sum = 0;
		float weightAccum = 0;

		float centerDepth = s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X];

#pragma unroll
		for(int j = -GAUSSIAN_SPATIAL_FILTER_RADIUS; j <= GAUSSIAN_SPATIAL_FILTER_RADIUS; j++)
		{
			float depth = s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			if(depth > 0.001f)
			{
				float spatialWeight = __expf(-(centerDepth-depth)*(centerDepth-depth)*inv2sig_t);
				sum +=	spatialWeight*cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j] * depth;
				weightAccum += spatialWeight*cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j];
			}
		}
		d_Dest[i * ROWS_BLOCKDIM_X] = sum/weightAccum;//Normalize
	}
}

__global__ void bilateralFilterKernelCols(Float3SOAPyramid vmapSOA, int xRes, int yRes, rgbd::framework::Intrinsics intr, float inv2sig_t)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	float* d_x = vmapSOA.x[0] + baseY * xRes + baseX;
	float* d_y = vmapSOA.y[0] + baseY * xRes + baseX;
	float* d_z = vmapSOA.z[0] + baseY * xRes + baseX;

	//Main data
#pragma unroll
	for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_z[i * COLUMNS_BLOCKDIM_Y * xRes];

	//Upper halo
	for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 
		(baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_z[i * COLUMNS_BLOCKDIM_Y * xRes] : 0;

	//Lower halo
	for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
		(yRes - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_z[i * COLUMNS_BLOCKDIM_Y * xRes] : 0;

	//End load step

	__syncthreads();

	//Compute and store results
#pragma unroll
	for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
		float sum = 0;
		float weightAccum = 0;

		float centerDepth = s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y];

#pragma unroll
		for(int j = -GAUSSIAN_SPATIAL_FILTER_RADIUS; j <= GAUSSIAN_SPATIAL_FILTER_RADIUS; j++)
		{
			float depth = s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			if(depth > 0.001f){
				//Compute bilateral filter weight
				float spatialWeight = __expf(-(centerDepth-depth)*(centerDepth-depth)*inv2sig_t);

				sum += spatialWeight * cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j] * depth;
				weightAccum += spatialWeight * cGaussianSpatialKernel[GAUSSIAN_SPATIAL_FILTER_RADIUS - j];
			}
		}

		float x = CUDART_NAN_F;
		float y = CUDART_NAN_F;
		float z = sum/weightAccum;//Normalize
		if (z > 0.001f) {//Exclude zero or negative depths.  
			int u = threadIdx.x+blockIdx.x*COLUMNS_BLOCKDIM_X;//Pixel space x coord
			int v = threadIdx.y+(blockIdx.y*COLUMNS_RESULT_STEPS + (i-COLUMNS_HALO_STEPS))*COLUMNS_BLOCKDIM_Y;//pixel space y coord
			x = (u - intr.cx) * z / intr.fx;
			y = (v - intr.cy) * z / intr.fy;
		} else{
			z = CUDART_NAN_F;
		}
		d_x[i * COLUMNS_BLOCKDIM_Y * xRes] = x;
		d_y[i * COLUMNS_BLOCKDIM_Y * xRes] = y;
		d_z[i * COLUMNS_BLOCKDIM_Y * xRes] = z;
	}

}


__host__ void buildVMapBilateralFilterCUDA(rgbd::framework::DPixel* dev_depthBuffer, Float3SOAPyramid vmapSOA, int xRes, int yRes, 
										  rgbd::framework::Intrinsics intr, float maxDepth, float sigma_t)
{
	//=====Row filter and maxdepth thresholding step======
	//Assert that kernel parameters are properly memory aligned. Otherwise, kernel will either fail or be inefficient
	assert(GAUSSIAN_SPATIAL_FILTER_RADIUS <= ROWS_BLOCKDIM_X*ROWS_HALO_STEPS);
	assert( xRes % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
	assert( yRes % ROWS_BLOCKDIM_Y == 0 );

	dim3 blocks(xRes / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), yRes / ROWS_BLOCKDIM_Y);
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	bilateralFilterKernelRows<<<blocks, threads>>>(dev_depthBuffer, vmapSOA, xRes, yRes, maxDepth, 1.0f/(2.0f*sigma_t));

	//======Column filter and point cloud generation======
	assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= GAUSSIAN_SPATIAL_FILTER_RADIUS );
    assert( xRes % COLUMNS_BLOCKDIM_X == 0 );
    assert( yRes % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    blocks = dim3(xRes / COLUMNS_BLOCKDIM_X, yRes / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    threads = dim3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	bilateralFilterKernelCols<<<blocks, threads>>>(vmapSOA, xRes, yRes, intr, 1.0f/(2.0f*sigma_t));

}

#pragma endregion

#pragma region RGB Map Generation

__global__ void rgbAOSToSOAKernel(rgbd::framework::ColorPixel* dev_colorPixels, 
								  Float3SOAPyramid rgbSOA, int xRes, int yRes)
{
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(u < xRes && v < yRes) 
	{
		int i = (v * xRes) + u;
		int i2 = (v * xRes) + (xRes - 1 - u);
		rgbd::framework::ColorPixel color = dev_colorPixels[i];
		rgbSOA.x[0][i2] = color.r / 255.0f;
		rgbSOA.y[0][i2] = color.g / 255.0f;
		rgbSOA.z[0][i2] = color.b / 255.0f;
	}

}

__host__ void rgbAOSToSOACUDA(rgbd::framework::ColorPixel* dev_colorPixels, 
							  Float3SOAPyramid rgbSOA, int xRes, int yRes)
{
	int tileSize = 16;

	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(xRes)/float(tileSize)), 
		(int)ceil(float(yRes)/float(tileSize)));

	rgbAOSToSOAKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_colorPixels, rgbSOA, xRes, yRes);
}



#pragma endregion

#pragma region VMap Subsampling
//Subsamples float3 SOA by 1/2 and stores in dest
//Threads are parallel by `
__global__ void subsampleFloat3Kernel(float* x_src, float* y_src, float* z_src, 
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

__host__ void subsamplePyramidCUDA(Float3SOAPyramid dev_mapSOA, int xRes, int yRes, int numLevels)
{
	int tileSize = 16;

	for(int i = 0; i < numLevels - 1; ++i)
	{
		dim3 threadsPerBlock(tileSize, tileSize);
		dim3 fullBlocksPerGrid((int)ceil(float(xRes>>(1+i))/float(tileSize)), 
			(int)ceil(float(yRes>>(1+i))/float(tileSize)));


		subsampleFloat3Kernel<<<fullBlocksPerGrid,threadsPerBlock>>>(dev_mapSOA.x[i], dev_mapSOA.y[i], dev_mapSOA.z[i],
			dev_mapSOA.x[i+1], dev_mapSOA.y[i+1], dev_mapSOA.z[i+1],
			xRes>>i, yRes>>i);
	}

}

#pragma endregion
