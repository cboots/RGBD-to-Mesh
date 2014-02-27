#include "seperable_filter.h"

#define SEPERABLE_KERNEL_RADIUS 3
#define SEPERABLE_KERNEL_SIZE 	(2*SEPERABLE_KERNEL_RADIUS+1)

__constant__ float cKernel[SEPERABLE_KERNEL_SIZE];

__host__ void setSeperableKernelGaussian(float sigma)
{
	float kernel[SEPERABLE_KERNEL_SIZE];

	for(int i = -SEPERABLE_KERNEL_RADIUS; i <= SEPERABLE_KERNEL_RADIUS; ++i)
	{
		kernel[i+SEPERABLE_KERNEL_RADIUS] = expf(-i*i/(2*sigma));
	}

	cudaMemcpyToSymbol(cKernel, kernel, SEPERABLE_KERNEL_RADIUS*sizeof(float));
}

__host__ void setSeperableUniform()
{

	float kernel[SEPERABLE_KERNEL_SIZE];

	for(int i = -SEPERABLE_KERNEL_RADIUS; i <= SEPERABLE_KERNEL_RADIUS; ++i)
	{
		kernel[i+SEPERABLE_KERNEL_RADIUS] = 1.0f;
	}

	cudaMemcpyToSymbol(cKernel, kernel, SEPERABLE_KERNEL_RADIUS*sizeof(float));
}

#pragma region Seperable Kernel

//Block width and height. 16 is convenient for memory alignment (4*16==64Bytes)
#define		ROWS_BLOCKDIM_X		16
#define		ROWS_BLOCKDIM_Y		12

//Number of pixels processed per thread
#define		ROWS_RESULT_STEPS	4
//Number of blocks in region. ROWS_HALO_STEPS*ROWS_BLOCKDIM_X must be > KERNEL_RADIUS
#define		ROWS_HALO_STEPS		1


__global__ void seperableKernelRows(float* x, float* y, float* z, float* x_out, float* y_out, float* z_out, 
									int xRes, int yRes)
{
	__shared__ float s_Data[3][ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X*(ROWS_RESULT_STEPS+2*ROWS_HALO_STEPS)];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;


	//Align source pointer
	x += baseY*xRes + baseX;//Align source pointer with this thread for convenience
	y += baseY*xRes + baseX;//Align source pointer with this thread for convenience
	z += baseY*xRes + baseX;//Align source pointer with this thread for convenience

	//Align output pointer
	x_out += baseY*xRes + baseX;//Align dest pointer with this thread for convenience
	y_out += baseY*xRes + baseX;//Align dest pointer with this thread for convenience
	z_out += baseY*xRes + baseX;//Align dest pointer with this thread for convenience


	//Main data
#pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; ++i)
	{
		s_Data[0][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = x[i*ROWS_BLOCKDIM_X];
		s_Data[1][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = y[i*ROWS_BLOCKDIM_X];
		s_Data[2][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = z[i*ROWS_BLOCKDIM_X];
	}

	//Left halo
	for(int i = 0; i < ROWS_HALO_STEPS; ++i)
	{
		if(baseX >= -i*ROWS_BLOCKDIM_X){
			s_Data[0][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = x[i*ROWS_BLOCKDIM_X];
			s_Data[1][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = y[i*ROWS_BLOCKDIM_X];
			s_Data[2][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = z[i*ROWS_BLOCKDIM_X];
		}else{
			s_Data[0][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0.0f;
			s_Data[1][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0.0f;
			s_Data[2][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0.0f;
		}

	}

	//Right halo
	for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; ++i)
	{
		if(xRes - baseX > i*ROWS_BLOCKDIM_X){
			s_Data[0][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = x[i*ROWS_BLOCKDIM_X];
			s_Data[1][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = y[i*ROWS_BLOCKDIM_X];
			s_Data[2][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = z[i*ROWS_BLOCKDIM_X];
		}else{
			s_Data[0][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0.0f;
			s_Data[1][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0.0f;
			s_Data[2][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 0.0f;
		}
	}
	__syncthreads();


	//=======END OF LOADING STAGE======

	//=======BEGIN COMPURE STAGE=======
#pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){//For each result block
		float sumX = 0;
		float sumY = 0;
		float sumZ = 0;
		float weightAccum = 0;

#pragma unroll
		for(int j = -SEPERABLE_KERNEL_RADIUS; j <= SEPERABLE_KERNEL_RADIUS; j++)
		{
			float weight = cKernel[SEPERABLE_KERNEL_RADIUS - j];
			weightAccum += weight;
			float x = s_Data[0][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			float y = s_Data[1][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			float z = s_Data[2][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			sumX +=	weight * x;
			sumY +=	weight * y;
			sumZ +=	weight * z;
		}
		x_out[i * ROWS_BLOCKDIM_X] = sumX/weightAccum;//Normalize
		y_out[i * ROWS_BLOCKDIM_X] = sumY/weightAccum;//Normalize
		z_out[i * ROWS_BLOCKDIM_X] = sumZ/weightAccum;//Normalize
	}
}


//=========SIMPLE GAUSSIAN KERNEL COLUMNS========
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1

__global__ void seperableKernelCols(float* x, float* y, float* z, float* x_out, float* y_out, float* z_out, 
								   int xRes, int yRes)
{
	__shared__ float s_Data[3][COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

	//Align source pointer
	x += baseY*xRes + baseX;//Align source pointer with this thread for convenience
	y += baseY*xRes + baseX;//Align source pointer with this thread for convenience
	z += baseY*xRes + baseX;//Align source pointer with this thread for convenience

	//Align output pointer
	x_out += baseY*xRes + baseX;//Align dest pointer with this thread for convenience
	y_out += baseY*xRes + baseX;//Align dest pointer with this thread for convenience
	z_out += baseY*xRes + baseX;//Align dest pointer with this thread for convenience



	//Main data
#pragma unroll
	for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		s_Data[0][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = x[i * COLUMNS_BLOCKDIM_Y * xRes];
		s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = y[i * COLUMNS_BLOCKDIM_Y * xRes];
		s_Data[2][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = z[i * COLUMNS_BLOCKDIM_Y * xRes];
	}

	//Upper halo
	for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		if(baseY >= -i * COLUMNS_BLOCKDIM_Y)
		{
			s_Data[0][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = x[i * COLUMNS_BLOCKDIM_Y * xRes];
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = y[i * COLUMNS_BLOCKDIM_Y * xRes];
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = z[i * COLUMNS_BLOCKDIM_Y * xRes];
		}else
		{
			s_Data[0][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0.0f;
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0.0f;
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0.0f;
		}
	}

	//Lower halo
	for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		if(yRes - baseY > i * COLUMNS_BLOCKDIM_Y)
		{
			s_Data[0][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = x[i * COLUMNS_BLOCKDIM_Y * xRes];
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = y[i * COLUMNS_BLOCKDIM_Y * xRes];
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = z[i * COLUMNS_BLOCKDIM_Y * xRes];
		}else
		{
			s_Data[0][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0.0f;
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0.0f;
			s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 0.0f;
		}
	}

	//End load step

	__syncthreads();

	//Compute and store results
#pragma unroll
	for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
		
		float sumX = 0;
		float sumY = 0;
		float sumZ = 0;
		float weightAccum = 0;

#pragma unroll
		for(int j = -SEPERABLE_KERNEL_RADIUS; j <= SEPERABLE_KERNEL_RADIUS; j++)
		{
			float weight = cKernel[SEPERABLE_KERNEL_RADIUS - j];
			weightAccum += weight;
			float x = s_Data[0][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			float y = s_Data[1][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			float z = s_Data[2][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			sumX +=	weight * x;
			sumY +=	weight * y;
			sumZ +=	weight * z;
		}
		x_out[i * ROWS_BLOCKDIM_X] = sumX/weightAccum;//Normalize
		y_out[i * ROWS_BLOCKDIM_X] = sumY/weightAccum;//Normalize
		z_out[i * ROWS_BLOCKDIM_X] = sumZ/weightAccum;//Normalize
	}

}



__host__ void seperableFilter(float* x, float* y, float* z, float* x_out, float* y_out, float* z_out, 
								   int xRes, int yRes)
{
	//=====Row filter step======
	//Assert that kernel parameters are properly memory aligned. Otherwise, kernel will either fail or be inefficient
	assert(SEPERABLE_KERNEL_RADIUS <= ROWS_BLOCKDIM_X*ROWS_HALO_STEPS);
	assert( xRes % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
	assert( yRes % ROWS_BLOCKDIM_Y == 0 );

	dim3 blocks(xRes / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), yRes / ROWS_BLOCKDIM_Y);
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	seperableKernelRows<<<blocks, threads>>>(x,y,z,x_out, y_out, z_out, xRes, yRes);

	//======Column filter step======
	assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= SEPERABLE_KERNEL_RADIUS );
    assert( xRes % COLUMNS_BLOCKDIM_X == 0 );
    assert( yRes % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    blocks = dim3(xRes / COLUMNS_BLOCKDIM_X, yRes / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    threads = dim3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	seperableKernelCols<<<blocks, threads>>>(x_out,y_out,z_out,x_out, y_out, z_out, xRes, yRes);
}