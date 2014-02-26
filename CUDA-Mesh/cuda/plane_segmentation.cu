#include "plane_segmentation.h"

/*
#pragma region Minimum Curvature Calculation
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
*/