#include "scan.h"

#define MAX_BLOCK_SIZE 1024
#define MAX_GRID_SIZE 65535

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5 

#define NO_BANK_CONFLICTS


#ifdef NO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n)    \
	(((n) >> (2 * LOG_NUM_BANKS)))  
#else
#define CONFLICT_FREE_OFFSET(a)    (0)  
#endif


inline int pow2roundup (int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}


__global__ void exclusive_scan_kernel(float* dev_in, float* dev_out, int width, int height)
{
	extern __shared__ float temp[];

	//Offset pointers to this block's row. Avoids the need for more complex indexing
	dev_in += width*blockIdx.x;
	dev_out += width*blockIdx.x;

	//Now each row is working with it's own row like a normal exclusive scan of an array length width.
	int index = threadIdx.x;
	int offset = 1;
	int n = 2*blockDim.x;//get actual temp padding
	
	int ai = index;
	int bi = index + n/2;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	//Bounds checking, load shared mem
	temp[ai+bankOffsetA] = (ai < width)?dev_in[ai]:0;
	temp[bi+bankOffsetB] = (bi < width)?dev_in[bi]:0;
	
	//Reduction step
	for (int d = n>>1; d > 0; d >>= 1)                  
	{   
		__syncthreads();  //Make sure previous step has completed
		if (index < d)  
		{
			int ai2 = offset*(2*index+1)-1;  
			int bi2 = offset*(2*index+2)-1;  
			ai2 += CONFLICT_FREE_OFFSET(ai2);
			bi2 += CONFLICT_FREE_OFFSET(bi2);

			temp[bi2] += temp[ai2];
		}  
		offset *= 2;  //Adjust offset
	}

	//Reduction complete

	//Clear last element
	if(index == 0)
		temp[(n-1)+CONFLICT_FREE_OFFSET(n-1)] = 0;


	//Sweep down
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{  
		offset >>= 1;  
		__syncthreads();  //wait for previous step to finish
		if (index < d)                       
		{  
			int ai2 = offset*(2*index+1)-1;  
			int bi2 = offset*(2*index+2)-1;  
			ai2 += CONFLICT_FREE_OFFSET(ai2);
			bi2 += CONFLICT_FREE_OFFSET(bi2);

			//Swap
			float t = temp[ai2];  
			temp[ai2] = temp[bi2];  
			temp[bi2] += t;   
		}  
	}  

	//Sweep complete
	__syncthreads();

	//Writeback
	if(ai < width)
		dev_out[ai] = temp[ai+bankOffsetA];
	if(bi < width)
		dev_out[bi] = temp[bi+bankOffsetB];


}

__host__ void exclusiveScanRows(float* dev_in, float* dev_out, int width, int height)
{
	//Make sure matrix is limits of this kernel.
	//Other algorithms can get around these limits, but this algorithm is simplified for the expected size of 640*480 
	assert(width <= 1024);
	assert(height <= MAX_GRID_SIZE);

	//Nearest power of two below width
	int blockArraySize = pow2roundup(width);
	dim3 threads(blockArraySize >> 1);//2 elements per thread
	dim3 blocks(height);
	int sharedCount = (2*blockArraySize+2)*sizeof(float);

	exclusive_scan_kernel<<<blocks,threads,sharedCount>>>(dev_in, dev_out, width, height);
}