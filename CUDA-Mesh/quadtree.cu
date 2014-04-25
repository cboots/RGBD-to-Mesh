#include "quadtree.h"

//R2 = R1*multFactor + R2;
__device__ void add_r1_to_r2(glm::mat3 &A, glm::vec3 &b, int r1, int r2, float multFactor)
{
	float tmp;

	if (r1 == r2) return;

	for(int i = 0; i < 3; ++i)
	{
		A[i][r2] = A[i][r2] + multFactor*A[i][r1];
	}

	b[r2] = b[r2] + multFactor*b[r1];

}

__device__ void swap_row(glm::mat3 &A, glm::vec3 &b, int r1, int r2)
{
	float tmp;

	if (r1 == r2) return;

#pragma unroll
	for(int i = 0; i < 3; ++i)
	{
		tmp = A[i][r1];
		A[i][r1] = A[i][r2];
		A[i][r2] = tmp;
	}

	tmp = b[r1];
	b[r1] = b[r2];
	b[r2] = tmp;

}

__device__ void row_mult(glm::mat3 &A, glm::vec3 &b, int r1, float mult)
{
#pragma unroll
	for(int i = 0; i < 3; ++i)
	{
		A[i][r1] = A[i][r1]*mult;
	}

	b[r1] *= mult;

}


#define APPROXZERO(a) (abs(a) < 0.000001f)
__device__ void makeNonZeroDiagonal(glm::mat3 &A,glm::vec3 b)
{
	int permute[3];

	if(		 !APPROXZERO(A[0][0]) && !APPROXZERO(A[1][1]) && !APPROXZERO(A[2][2]))
	{
		permute[0] = 0; permute[1] = 1; permute[2] = 2;
	}else if(!APPROXZERO(A[0][0]) && !APPROXZERO(A[1][2]) && !APPROXZERO(A[2][1]))
	{
		permute[0] = 0; permute[1] = 2; permute[2] = 1;
	}else if(!APPROXZERO(A[0][1]) && !APPROXZERO(A[1][0]) && !APPROXZERO(A[2][2]))
	{
		permute[0] = 1; permute[1] = 0; permute[2] = 2;
	}else if(!APPROXZERO(A[0][1]) && !APPROXZERO(A[1][2]) && !APPROXZERO(A[2][0]))
	{
		permute[0] = 1; permute[1] = 2; permute[2] = 0;
	}else if(!APPROXZERO(A[0][2]) && !APPROXZERO(A[1][0]) && !APPROXZERO(A[2][1]))
	{
		permute[0] = 2; permute[1] = 0; permute[2] = 1;
	}else if(!APPROXZERO(A[0][2]) && !APPROXZERO(A[1][1]) && !APPROXZERO(A[2][0]))
	{
		permute[0] = 2; permute[1] = 1; permute[2] = 0;
	}else{
		//ERROR
	}

	for(int i = 0; i < 3; ++i)
	{
		if(permute[i] > i)
		{
			swap_row(A,b, i, permute[i]);
			for(int j = 0; j < 3; ++j)
			{
				if(permute[j] == i)
				{
					permute[j] = permute[i];
					permute[i] = i;
					break;
				}
			}
		}
	}
}

__device__ glm::vec3 solveAbGaussian(glm::mat3 A, glm::vec3 b)
{
	//Make sure diagonals have non-zero entries
	if(abs(A[0][0]*A[1][1]*A[2][2]) < 0.000001f)
		makeNonZeroDiagonal(A,b);

	//Row echelon form
	for(int r = 0; r < 3; ++r)
	{
		float factor = 1.0f/A[r][r];
		row_mult(A,b,r,factor);
		for(int r2 = r+1; r2 < 3; ++r2)
		{
			if(abs(A[r][r2]) > 0.000001f)
			{
				//If A[r][r2] not zero yet, 
				//Need A[r][r2] + factor*A[r][r] == 0
				factor = -A[r][r2]/A[r][r]; 
				add_r1_to_r2(A,b,r,r2,factor);
			}
		}
	}

	//Matrix now upper triangular
	//Back substitute
	for(int r = 0; r < 3; ++r)
	{
		for(int c = r+1; c < 3; ++c)
		{
			if(abs(A[c][r]) > 0.000001f)
			{
				//element is non-zero. Backsubstitute
				//Need A[c][r] + factor*A[c][c] == 0
				float factor = -A[c][r]/A[c][c];
				add_r1_to_r2(A,b,c,r,factor);
			}
		}
	}
	return b;

}



//Numthreads is assumed to be a power of two
__device__ void minmaxreduction(float* s_minSx, float* s_maxSx, float* s_minSy, float* s_maxSy, int indexInBlock, int nTotalThreads)
{
	int  thread2;
	float temp;

	while(nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (indexInBlock < halfPoint)
		{
			thread2 = indexInBlock + halfPoint;

			// Get the shared value stored by another thread
			temp = s_minSx[thread2];
			if (temp < s_minSx[indexInBlock]) 
				s_minSx[indexInBlock] = temp; 

			temp = s_minSy[thread2];
			if (temp < s_minSy[indexInBlock]) 
				s_minSy[indexInBlock] = temp; 

			temp = s_maxSx[thread2];
			if (temp > s_maxSx[indexInBlock]) 
				s_maxSx[indexInBlock] = temp; 

			temp = s_maxSy[thread2];
			if (temp > s_maxSy[indexInBlock]) 
				s_maxSy[indexInBlock] = temp; 
		}
		__syncthreads();

		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
}

__global__ void computeAABBsKernel(PlaneStats* planeStats, int* planeInvIdMap, glm::vec4* aabbsBlockResults,
								   int* planeCount, int maxPlanes,
								   Float3SOA positions, float* segmentProjectedSx, float* segmentProjectedSy, 
								   int* finalSegmentsBuffer, int xRes, int yRes)
{
	extern __shared__ int s_Mem[];
	int* s_InvMap = (int*) s_Mem;
	float* s_centroidX = (float*)(s_InvMap + maxPlanes);
	float* s_centroidY = s_centroidX + maxPlanes;
	float* s_centroidZ = s_centroidY + maxPlanes;
	glm::vec3* s_tangents = (glm::vec3*) (s_centroidZ + maxPlanes);
	glm::vec3* s_bitangents = s_tangents + maxPlanes;
	float* s_minSx = (float*)(s_bitangents + maxPlanes);
	float* s_minSy = (s_minSx + blockDim.x*blockDim.y);
	float* s_maxSx = (s_minSy + blockDim.x*blockDim.y);
	float* s_maxSy = (s_maxSx + blockDim.x*blockDim.y);

	int indexInBlock = threadIdx.x + threadIdx.y*blockDim.x;
	int imageX = threadIdx.x + blockDim.x*blockIdx.x;
	int imageY = threadIdx.y + blockDim.y*blockIdx.y;


	int numPlanes = planeCount[0];
	if(indexInBlock < maxPlanes)
	{
		s_InvMap[indexInBlock] = planeInvIdMap[indexInBlock];
		if(indexInBlock < numPlanes)
		{
			//s_aabb[indexInBlock] = glm::vec4(0.0f);
			s_tangents[indexInBlock] = planeStats[indexInBlock].tangent;
			s_centroidX[indexInBlock] = planeStats[indexInBlock].centroid.x;
			s_centroidY[indexInBlock] = planeStats[indexInBlock].centroid.y;
			s_centroidZ[indexInBlock] = planeStats[indexInBlock].centroid.z;
			//bitangent = norm cross tangent
			glm::vec3 norm(planeStats[indexInBlock].norm.x,planeStats[indexInBlock].norm.y,planeStats[indexInBlock].norm.z);
			s_bitangents[indexInBlock] = glm::normalize(glm::cross(norm, s_tangents[indexInBlock]));
		}
	}
	__syncthreads();

	//Remap segments
	int segment = finalSegmentsBuffer[imageX + imageY*xRes];
	float sx = 0;
	float sy = 0;
	if(segment >= 0)
	{
		//Remap and writeback
		segment = s_InvMap[segment];
		finalSegmentsBuffer[imageX + imageY*xRes] = segment;

		//Compute Sx and Sy
		glm::vec3 dp = glm::vec3(positions.x[imageX + imageY*xRes] - s_centroidX[segment], 
			positions.y[imageX + imageY*xRes] - s_centroidY[segment],
			positions.z[imageX + imageY*xRes] - s_centroidZ[segment]);

		sx = glm::dot(dp, s_bitangents[segment]);
		sy = glm::dot(dp, s_tangents[segment]);


	}
	//writeback
	segmentProjectedSx[imageX + imageY*xRes] = sx;
	segmentProjectedSy[imageX + imageY*xRes] = sy;

	__syncthreads();
	//Repurpose invmap sharedmem for segment flags

	if(indexInBlock < maxPlanes)
	{
		s_InvMap[indexInBlock] = 0;
	}

	__syncthreads();
	if(segment >= 0)//flag each segment that exists in this block
		s_InvMap[segment] = 1;

	for(int plane = 0; plane < numPlanes; ++plane)
	{
		if(s_InvMap[plane] > 0)
		{

			//Init minmax planes
			s_minSx[indexInBlock] = (segment == plane)?sx:0;
			s_maxSx[indexInBlock] = (segment == plane)?sx:0;
			s_minSy[indexInBlock] = (segment == plane)?sy:0;
			s_maxSy[indexInBlock] = (segment == plane)?sy:0;
			__syncthreads();
			minmaxreduction(s_minSx, s_maxSx, s_minSy, s_maxSy, indexInBlock, blockDim.x*blockDim.y);
			//Threads already synced in function

			if(indexInBlock == 0)
			{
				aabbsBlockResults[(blockIdx.x + blockIdx.y*gridDim.x) + plane*gridDim.x*gridDim.y] 
					= glm::vec4(s_minSx[0], s_maxSx[0],s_minSy[0],s_maxSy[0]);
			}
		}else{
			if(indexInBlock == 0)
				aabbsBlockResults[(blockIdx.x + blockIdx.y*gridDim.x) + plane*gridDim.x*gridDim.y] = glm::vec4(0.0f);
		}
	}
}


__global__ void reduceAABBsKernel(PlaneStats* planeStats, glm::vec4* aabbsBlockResults, int numBlocks, int maxPlanes, int* planeCount)
{
	extern __shared__ float s_temp[];
	float* s_minSx = s_temp;
	float* s_minSy = (s_minSx + blockDim.x);
	float* s_maxSx = (s_minSy + blockDim.x);
	float* s_maxSy = (s_maxSx + blockDim.x);

	//two elements loaded per thread
	int i = threadIdx.x;
	int i2 = threadIdx.x + blockDim.x;

	int numPlanes = planeCount[0];
	for(int plane = 0; plane < numPlanes; ++plane)
	{
		glm::vec4 aabb1(0.0f);
		glm::vec4 aabb2(0.0f);
		if(i < numBlocks)
			aabb1 = aabbsBlockResults[i + plane*numBlocks];
		if(i2 < numBlocks)
			aabb2 = aabbsBlockResults[i2 + plane*numBlocks];

		s_minSx[i] = MIN(aabb1.x,aabb2.x);
		s_maxSx[i] = MAX(aabb1.y,aabb2.y);
		s_minSy[i] = MIN(aabb1.z,aabb2.z);
		s_maxSy[i] = MAX(aabb1.w,aabb2.w);

		__syncthreads();
		minmaxreduction(s_minSx, s_maxSx, s_minSy, s_maxSy, i, blockDim.x);

		if(threadIdx.x == 0)
			planeStats[plane].projParams.aabbMeters = glm::vec4(s_minSx[0], s_maxSx[0],s_minSy[0],s_maxSy[0]);
	}
}


__host__ __device__ int roundupnextpow2 (int x)
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

__host__ void computeAABBs(PlaneStats* planeStats, int* planeInvIdMap, glm::vec4* aabbsBlockResults,
						   int* planeCount, int maxPlanes,
						   Float3SOA positions, float* segmentProjectedSx, float* segmentProjectedSy, 
						   int* finalSegmentsBuffer, int xRes, int yRes)
{
	int blockWidth = AABB_COMPUTE_BLOCKWIDTH;
	int blockHeight = AABB_COMPUTE_BLOCKHEIGHT;

	assert(blockHeight*blockWidth >= maxPlanes);
	dim3 threads(blockWidth, blockHeight);
	dim3 blocks((int) ceil(xRes/float(blockWidth)), (int) ceil(yRes/float(blockHeight)));
	//plane map, tangent, bitangent, centroid and aabb of each plane loaded to shared memory.
	int sharedMem = maxPlanes*(sizeof(int) + sizeof(float)*3+sizeof(glm::vec3)*2) + blockWidth*blockHeight*4*sizeof(float);

	computeAABBsKernel<<<blocks,threads,sharedMem>>>(planeStats, planeInvIdMap, aabbsBlockResults, planeCount, maxPlanes,
		positions, segmentProjectedSx, segmentProjectedSy, 
		finalSegmentsBuffer, xRes, yRes);

	int numBlocks = blocks.x*blocks.y;
	int pow2Blocks = roundupnextpow2 (numBlocks) >> 1;//Next lowest power of two
	assert(pow2Blocks <= 1024);


	threads = dim3(pow2Blocks);
	blocks = dim3(1);
	sharedMem = 4*sizeof(float)*pow2Blocks;
	reduceAABBsKernel<<<blocks,threads,sharedMem>>>(planeStats, aabbsBlockResults, numBlocks, maxPlanes, planeCount);

}


__global__ void calculateProjectionDataKernel(rgbd::framework::Intrinsics intr, PlaneStats* planeStats,
											  int* planeCount, int maxTextureSize, int xRes, int yRes)
{
	glm::mat3 C(1.0f);

	int destWidth  = 0;
	int destHeight = 0;
	int maxRatio = 0;
	glm::vec4 aabb;
	if(threadIdx.x < planeCount[0])
	{
		//In range and valid plane.

		glm::vec3 tangent = planeStats[threadIdx.x].tangent;
		glm::vec3 normal = glm::vec3(planeStats[threadIdx.x].norm.x,planeStats[threadIdx.x].norm.y,planeStats[threadIdx.x].norm.z);
		glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

		glm::vec3 centroid = glm::vec3(planeStats[threadIdx.x].centroid.x,
			planeStats[threadIdx.x].centroid.y,
			planeStats[threadIdx.x].centroid.z);
		aabb = planeStats[threadIdx.x].projParams.aabbMeters;

		//Compute camera space coordinates (4 points in clockwise winding from viewpoint)
		/*   1----2
		*    |    |
		*    4----3
		*/
		glm::vec3 sp1 = (aabb.x*bitangent)+(aabb.z*tangent)+centroid;//UL, Sxmin,Symin
		glm::vec3 sp2 = (aabb.y*bitangent)+(aabb.z*tangent)+centroid;//UR, Sxmax,Symin
		glm::vec3 sp3 = (aabb.y*bitangent)+(aabb.w*tangent)+centroid;//LR, Sxmax,Symax
		glm::vec3 sp4 = (aabb.x*bitangent)+(aabb.w*tangent)+centroid;//LL, Sxmin,Symax

		//Compute screen space projections
		float su1 = sp1.x*intr.fx/sp1.z + intr.cx;
		float sv1 = sp1.y*intr.fy/sp1.z + intr.cy;
		float su2 = sp2.x*intr.fx/sp2.z + intr.cx;
		float sv2 = sp2.y*intr.fy/sp2.z + intr.cy;
		float su3 = sp3.x*intr.fx/sp3.z + intr.cx;
		float sv3 = sp3.y*intr.fy/sp3.z + intr.cy;
		float su4 = sp4.x*intr.fx/sp4.z + intr.cx;
		float sv4 = sp4.y*intr.fy/sp4.z + intr.cy;

		//Compute desired resolution.
		float sourceWidthMeters = aabb.y-aabb.x;
		float sourceHeightMeters = aabb.w-aabb.z;

		//Compute minimum resolution for complete data preservation
		float d12 = sqrtf((su1-su2)*(su1-su2)+(sv1-sv2)*(sv1-sv2));
		float d23 = sqrtf((su2-su3)*(su2-su3)+(sv2-sv3)*(sv2-sv3));
		float d34 = sqrtf((su3-su4)*(su3-su4)+(sv3-sv4)*(sv3-sv4));
		float d41 = sqrtf((su4-su1)*(su4-su1)+(sv4-sv1)*(sv4-sv1));
		float maxXRatio = MAX(d12,d34)/sourceWidthMeters;
		float maxYRatio = MAX(d23,d41)/sourceHeightMeters;

		maxRatio = ceil(MAX(maxXRatio,maxYRatio));
		maxRatio = roundupnextpow2(maxRatio);

		destWidth  = maxRatio * sourceWidthMeters;
		destHeight = maxRatio * sourceHeightMeters;

		//Make sure it fits. If not, then scale down
		if(destWidth > maxTextureSize || destHeight > maxTextureSize)
		{
			int scale = glm::max(ceil(destWidth/float(maxTextureSize)),ceil(destHeight/float(maxTextureSize)));
			scale = roundupnextpow2(scale);
			destWidth/=scale;
			destHeight/=scale;

		}

		//Compute A matrix (source points to basis vectors)
		glm::mat3 A = glm::mat3(su1,sv1,1,su2,sv2,1,su3,sv3,1);
		glm::vec3 b = glm::vec3(su4,sv4, 1);
		glm::vec3 x = glm::inverse(A)*b; 
		//mult each row i by xi
		for(int i = 0; i < 3; ++i)
		{
			A[i][0] *= x[i];
			A[i][1] *= x[i];
			A[i][2] *= x[i];
		}


		//Compute B matrix (dest points to basis vectors)
		glm::mat3 B = glm::mat3(0,0,1,
			destWidth,0,1,
			destWidth,destHeight,1);
		b = glm::vec3(0,destHeight, 1);

		x = glm::inverse(B)*b;
		//mult each row i by xi
		for(int i = 0; i < 3; ++i)
		{
			B[i][0] *= x[i];
			B[i][1] *= x[i];
			B[i][2] *= x[i];
		}

		C = A*glm::inverse(B);

		
	}

	planeStats[threadIdx.x].projParams.projectionMatrix = C;
	planeStats[threadIdx.x].projParams.aabbMeters = aabb;
	planeStats[threadIdx.x].projParams.destWidth = destWidth;
	planeStats[threadIdx.x].projParams.destHeight = destHeight;
	planeStats[threadIdx.x].projParams.textureResolution = maxRatio;
}


__host__ void calculateProjectionData(rgbd::framework::Intrinsics intr, PlaneStats* planeStats,
									  int* planeCount, int maxTextureSize, int maxPlanes, int xRes, int yRes)
{
	dim3 blocks(1);
	dim3 threads(maxPlanes);

	calculateProjectionDataKernel<<<blocks,threads>>>(intr, planeStats, planeCount, maxTextureSize, xRes, yRes);
}


__global__ void projectTexture(int segmentId, PlaneStats* dev_planeStats, 
							   Float4SOA destTexture, int destTextureSize, 
							   RGBMapSOA rgbMap, int* dev_finalSegmentsBuffer, float* dev_finalDistanceToPlaneBuffer,
							   int imageXRes, int imageYRes)
{
	int destX = blockIdx.x*blockDim.x+threadIdx.x;
	int destY = blockIdx.y*blockDim.y+threadIdx.y;

	if(destX < destTextureSize && destX < dev_planeStats->projParams.destWidth
		&& destY < destTextureSize && destY < dev_planeStats->projParams.destHeight)
	{
		float r = CUDART_NAN_F;
		float g = CUDART_NAN_F;
		float b = CUDART_NAN_F;
		float dist = CUDART_NAN_F;

		//Destination in range
		glm::mat3 Tds = dev_planeStats->projParams.projectionMatrix;

		glm::vec3 sourceCoords = Tds*glm::vec3(destX, destY, 1.0f);

		//Dehomogenization
		sourceCoords.x /= sourceCoords.z;
		sourceCoords.y /= sourceCoords.z;

		if(sourceCoords.x >= 0 && sourceCoords.x < imageXRes 
			&& sourceCoords.y >= 0 && sourceCoords.y < imageYRes )
		{
			//In source range
			int linIndex = int(sourceCoords.x) + int(sourceCoords.y)*imageXRes;
			if(segmentId == dev_finalSegmentsBuffer[linIndex]){
				r = rgbMap.r[linIndex];
				g = rgbMap.g[linIndex];
				b = rgbMap.b[linIndex];
				dist = dev_finalSegmentsBuffer[linIndex];
			}
		}

		destTexture.x[destX + destY*destTextureSize] = r;
		destTexture.y[destX + destY*destTextureSize] = g;
		destTexture.z[destX + destY*destTextureSize] = b;
		destTexture.w[destX + destY*destTextureSize] = dist;
	}

}


__host__ void projectTexture(int segmentId, PlaneStats* host_planeStats, PlaneStats* dev_planeStats,  
							 Float4SOA destTexture, int destTextureSize, 
							 RGBMapSOA rgbMap, int* dev_finalSegmentsBuffer, float* dev_finalDistanceToPlaneBuffer,
							 int imageXRes, int imageYRes)
{
	int tileSize = 16;

	dim3 threads(tileSize, tileSize);
	dim3 blocks((int)ceil(float(host_planeStats->projParams.destWidth)/float(tileSize)),
		(int)ceil(float(host_planeStats->projParams.destHeight)/float(tileSize)));

	projectTexture<<<blocks,threads>>>(segmentId, dev_planeStats, destTexture, destTextureSize, 
		rgbMap, dev_finalSegmentsBuffer, dev_finalDistanceToPlaneBuffer, imageXRes, imageYRes);
}


__global__ void quadtreeDecimationKernel1(int actualWidth, int actualHeight, Float4SOA planarTexture, int* quadTreeAssemblyBuffer,
										  int textureBufferSize)
{
	extern __shared__ int s_tile[];

	//======================Load==========================
	//Global index
	int gx = threadIdx.x + blockDim.x*blockIdx.x;
	int gy = threadIdx.y + blockDim.y*blockIdx.y;
	int s_index = threadIdx.x + threadIdx.y*(blockDim.x+1);
	int indexInBlock = threadIdx.x + threadIdx.y*blockDim.x;
	//Load shared memory
	//load core. If in range and texture buffer has valid pixel at this location, load 0. Else, load -1;
	int val = -1;
	if(gx < actualWidth && gy < actualHeight)
	{
		float pixelContents = planarTexture.x[gx+gy*textureBufferSize];
		if(pixelContents == pixelContents)
		{
			val = 0;//Pixel is valid point. save
		}
	}
	s_tile[s_index] = val;

	//Load apron
	if(indexInBlock < (blockDim.x*2+1))//first 33 threads load remaining apron
	{
		if(indexInBlock < blockDim.x)//first 16 load bottom
		{
			gx = indexInBlock  + blockDim.x*blockIdx.x;
			gy = blockDim.y*(blockIdx.y+1);//first row of next block
			s_index = indexInBlock + (blockDim.y*(blockDim.x+1));
		}else if(indexInBlock < blockDim.x*2){//next 16 load right apron
			gx = blockDim.x*(blockIdx.x+1);//First column of next block
			gy = blockDim.y*blockIdx.y + (indexInBlock % blockDim.x);//indexInBlock % blockDim.x is y position in block
			s_index = blockDim.x + ((indexInBlock % blockDim.x)*(blockDim.x+1));
		}else{
			//load the corner
			gx = blockDim.x*(blockIdx.x+1);
			gy = blockDim.y*(blockIdx.y+1);
			s_index = blockDim.x + blockDim.y*(blockDim.x+1);
		}

		val = -1;
		if(gx < actualWidth && gy < actualHeight)
		{
			float pixelContents = planarTexture.x[gx+gy*textureBufferSize];
			if(pixelContents == pixelContents)
			{
				val = 0;//Pixel is valid point. save
			}

		}
		s_tile[s_index] = val;
	}
	__syncthreads();

	//====================Reduction=========================


	//Step == 0 is special case. need to initialize baseline quads
	bool merge = false;
	if(s_tile[threadIdx.x+threadIdx.y*(blockDim.x+1)] == 0)
	{
		//Check neighbors. If all neighbors right down and right-down diagonal are 0, set to 1.
		if(		s_tile[(threadIdx.x+1)	+	(threadIdx.y)  *(blockDim.x+1)] == 0
			&&	s_tile[(threadIdx.x  )	+	(threadIdx.y+1)*(blockDim.x+1)] == 0
			&&	s_tile[(threadIdx.x+1)	+	(threadIdx.y+1)*(blockDim.x+1)] == 0)
		{
			merge = true;
		}
	}

	__syncthreads();
	if(merge)
	{
		s_tile[threadIdx.x+threadIdx.y*(blockDim.x+1)] = 1;
	}

	__syncthreads();


	//Loop for remaining steps
	for(int step = 1; step < blockDim.x; step <<= 1)
	{
		if((threadIdx.x % (step*2)) == 0 && (threadIdx.y % (step*2)) == 0)
		{
			//Corner points only.
			if(	s_tile[(threadIdx.x)	+	(threadIdx.y)  *(blockDim.x+1)] == step
				&&  s_tile[(threadIdx.x+step)	+	(threadIdx.y)	  *(blockDim.x+1)] == step
				&&	s_tile[(threadIdx.x		)	+	(threadIdx.y+step)*(blockDim.x+1)] == step
				&&	s_tile[(threadIdx.x+step)	+	(threadIdx.y+step)*(blockDim.x+1)] == step)
			{
				//Upgrade degree of this point
				s_tile[(threadIdx.x)	+	(threadIdx.y)  *(blockDim.x+1)] *= 2;

				//Clear definitely removed points
				s_tile[(threadIdx.x+step)	+	(threadIdx.y)  *(blockDim.x+1)] = -1;
				s_tile[(threadIdx.x		)	+	(threadIdx.y+step)*(blockDim.x+1)] = -1;
				s_tile[(threadIdx.x+step)	+	(threadIdx.y+step)*(blockDim.x+1)] = -1;

			}
		}
		__syncthreads();

	}

	//====================Writeback=========================
	//writeback core.
	gx = threadIdx.x + blockDim.x*blockIdx.x;
	gy = threadIdx.y + blockDim.y*blockIdx.y;
	s_index = threadIdx.x + threadIdx.y*(blockDim.x+1);
	quadTreeAssemblyBuffer[gx+gy*textureBufferSize] = s_tile[s_index];

	//no need to writeback apron

}

__global__ void quadtreeDecimationKernel2(int actualWidth, int actualHeight, int* quadTreeAssemblyBuffer, int textureBufferSize)
{
	extern __shared__ int s_tile[];

	int scaleMultiplier = blockDim.x;

	//======================Load==========================
	//Global index (scaled by multiplier)
	int gx = scaleMultiplier*(threadIdx.x + blockDim.x*blockIdx.x);
	int gy = scaleMultiplier*(threadIdx.y + blockDim.y*blockIdx.y);
	int s_index = threadIdx.x + threadIdx.y*(blockDim.x+1);
	int indexInBlock = threadIdx.x + threadIdx.y*blockDim.x;
	//Load shared memory
	//load core. If in range and texture buffer has valid pixel at this location, load 0. Else, load -1;
	int val = -1;
	if(gx < actualWidth && gy < actualHeight)
	{
		val = quadTreeAssemblyBuffer[gx+gy*textureBufferSize];
	}
	s_tile[s_index] = val;

	//Load apron
	if(indexInBlock < (blockDim.x*2+1))//first 33 threads load remaining apron
	{
		if(indexInBlock < blockDim.x)//first 16 load bottom
		{
			gx = indexInBlock  + blockDim.x*blockIdx.x;
			gy = blockDim.y*(blockIdx.y+1);//first row of next block
			s_index = indexInBlock + (blockDim.y*(blockDim.x+1));
		}else if(indexInBlock < blockDim.x*2){//next 16 load right apron
			gx = blockDim.x*(blockIdx.x+1);//First column of next block
			gy = blockDim.y*blockIdx.y + (indexInBlock % blockDim.x);//indexInBlock % blockDim.x is y position in block
			s_index = blockDim.x + ((indexInBlock % blockDim.x)*(blockDim.x+1));
		}else{
			//load the corner
			gx = blockDim.x*(blockIdx.x+1);
			gy = blockDim.y*(blockIdx.y+1);
			s_index = blockDim.x + blockDim.y*(blockDim.x+1);
		}
		gx *= scaleMultiplier;
		gy *= scaleMultiplier;

		val = -1;
		if(gx < actualWidth && gy < actualHeight)
		{
			val = quadTreeAssemblyBuffer[gx+gy*textureBufferSize];
		}
		s_tile[s_index] = val;
	}
	__syncthreads();

	//====================Reduction=========================


	//Step == 0 is special case. need to initialize baseline quads
	bool merge = false;
	if(s_tile[threadIdx.x+threadIdx.y*(blockDim.x+1)] == 0)
	{
		//Check neighbors. If all neighbors right down and right-down diagonal are 0, set to 1.
		if(		s_tile[(threadIdx.x+1)	+	(threadIdx.y)  *(blockDim.x+1)] == 0
			&&	s_tile[(threadIdx.x  )	+	(threadIdx.y+1)*(blockDim.x+1)] == 0
			&&	s_tile[(threadIdx.x+1)	+	(threadIdx.y+1)*(blockDim.x+1)] == 0)
		{
			merge = true;
		}
	}

	__syncthreads();
	if(merge)
	{
		s_tile[threadIdx.x+threadIdx.y*(blockDim.x+1)] = 1;
	}

	__syncthreads();


	//Loop for remaining steps
	for(int step = 1; step < blockDim.x; step <<= 1)
	{
		if((threadIdx.x % (step*2)) == 0 && (threadIdx.y % (step*2)) == 0)
		{
			//Corner points only.
			if(	s_tile[(threadIdx.x)	+	(threadIdx.y)  *(blockDim.x+1)] == step*scaleMultiplier
				&&  s_tile[(threadIdx.x+step)	+	(threadIdx.y)	  *(blockDim.x+1)] == step*scaleMultiplier
				&&	s_tile[(threadIdx.x		)	+	(threadIdx.y+step)*(blockDim.x+1)] == step*scaleMultiplier
				&&	s_tile[(threadIdx.x+step)	+	(threadIdx.y+step)*(blockDim.x+1)] == step*scaleMultiplier)
			{
				//Upgrade degree of this point
				s_tile[(threadIdx.x)	+	(threadIdx.y)  *(blockDim.x+1)] *= 2;

				//Clear definitely removed points
				s_tile[(threadIdx.x+step)	+	(threadIdx.y)  *(blockDim.x+1)] = -1;
				s_tile[(threadIdx.x		)	+	(threadIdx.y+step)*(blockDim.x+1)] = -1;
				s_tile[(threadIdx.x+step)	+	(threadIdx.y+step)*(blockDim.x+1)] = -1;

			}
		}
		__syncthreads();

	}

	//====================Writeback=========================
	//writeback core.
	gx = scaleMultiplier*(threadIdx.x + blockDim.x*blockIdx.x);
	gy = scaleMultiplier*(threadIdx.y + blockDim.y*blockIdx.y);
	s_index = threadIdx.x + threadIdx.y*(blockDim.x+1);
	quadTreeAssemblyBuffer[gx+gy*textureBufferSize] = s_tile[s_index];

	//no need to writeback apron

}

__host__ void quadtreeDecimation(int actualWidth, int actualHeight, Float4SOA planarTexture, int* quadTreeAssemblyBuffer,
								 int textureBufferSize)
{
	//do two simplification passes. Max quadtree size will therefore be 2*tileSize
	int tileSize = 16;

	//Pass one, parallel by pixel
	dim3 threads(tileSize, tileSize);
	dim3 blocks((int)ceil(actualWidth/float(tileSize)),
		(int)ceil(actualHeight/float(tileSize)));
	int sharedSize = (tileSize+1)*(tileSize+1)*sizeof(int);
	quadtreeDecimationKernel1<<<blocks,threads,sharedSize>>>(actualWidth, actualHeight, planarTexture, quadTreeAssemblyBuffer, textureBufferSize);

	blocks = dim3((int)ceil(actualWidth/float(tileSize*tileSize)),
		(int)ceil(actualHeight/float(tileSize*tileSize)));
	quadtreeDecimationKernel2<<<blocks,threads,sharedSize>>>(actualWidth, actualHeight, quadTreeAssemblyBuffer, textureBufferSize);

}



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


__global__ void quadTreeExclusiveScanKernel(int width, int* input, int* output,  int bufferStride, int* blockResults)
{
	extern __shared__ float temp[];

	//Offset pointers to this block's row. Avoids the need for more complex indexing
	input += bufferStride*blockIdx.x;
	output += bufferStride*blockIdx.x;

	//Now each row is working with it's own row like a normal exclusive scan of an array length width.
	int index = threadIdx.x;
	int offset = 1;
	int n = 2*blockDim.x;//get actual temp padding

	int ai = index;
	int bi = index + n/2;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	//Bounds checking, load shared mem
	temp[ai+bankOffsetA] = (ai < width)?input[ai]:0;
	temp[bi+bankOffsetB] = (bi < width)?input[bi]:0;
	//Negative vertecies are to be cleared
	if(temp[ai+bankOffsetA] < 0)
		temp[ai+bankOffsetA] = 0;
	if(temp[bi+bankOffsetB] < 0)
		temp[bi+bankOffsetB] = 0;


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
	{
		blockResults[blockIdx.x] = temp[(n-1)+CONFLICT_FREE_OFFSET(n-1)];
		temp[(n-1)+CONFLICT_FREE_OFFSET(n-1)] = 0;
	}

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
		output[ai] = temp[ai+bankOffsetA];
	if(bi < width)
		output[bi] = temp[bi+bankOffsetB];

}


__global__ void blockResultsExclusiveScanKernel(int* blockResults, int numBlocks, int* totalSumOut)
{
	extern __shared__ float temp[];

	//Now each row is working with it's own row like a normal exclusive scan of an array length width.
	int index = threadIdx.x;
	int offset = 1;
	int n = 2*blockDim.x;//get actual temp padding

	int ai = index;
	int bi = index + n/2;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	//Bounds checking, load shared mem
	temp[ai+bankOffsetA] = (ai < numBlocks)?blockResults[ai]:0;
	temp[bi+bankOffsetB] = (bi < numBlocks)?blockResults[bi]:0;

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
	{
		totalSumOut[0] = temp[(n-1)+CONFLICT_FREE_OFFSET(n-1)];
		temp[(n-1)+CONFLICT_FREE_OFFSET(n-1)] = 0;
	}

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
	if(ai < numBlocks)
		blockResults[ai] = temp[ai+bankOffsetA];
	if(bi < numBlocks)
		blockResults[bi] = temp[bi+bankOffsetB];


}



__global__ void reintegrateResultsKernel(int actualWidth, int textureBufferSize, 
										 int* quadTreeScanResults,  int* blockResults)
{
	int pixelX = threadIdx.x;
	int pixelY = blockIdx.x;

	if(pixelX < actualWidth)
	{
		quadTreeScanResults[pixelX + pixelY*textureBufferSize] += blockResults[pixelY];
	}
}


__global__ void scatterResultsKernel(glm::vec4 aabbMeters, int actualWidth, int actualHeight, 
									 int finalTextureWidth, int finalTextureHeight, int textureBufferSize, 
									 int* quadTreeAssemblyBuffer,  int* quadTreeScanResults,  
									 int* blockResults,  int* indexBuffer, float4* vertexBuffer)
{
	int pixelX = threadIdx.x;
	int pixelY = blockIdx.x;

	if(pixelX < actualWidth)
	{
		int degree = quadTreeAssemblyBuffer[pixelX + pixelY*textureBufferSize];//Load vertex degree

		//Only continue if this is a used vertex in the quadtree
		if(degree >= 0)
		{
			int vertNum = quadTreeScanResults[pixelX + pixelY*textureBufferSize];

			//Compute vertex info.
			float textureU = float(pixelX)/float(finalTextureWidth);
			float textureV = float(pixelY)/float(finalTextureHeight);

			//pixelX*(Sxmax-Sxmin)/actualWidth + Sxmin;
			float posX = (pixelX*(aabbMeters.y-aabbMeters.x))/float(actualWidth) + aabbMeters.x;
			//pixelY*(Symax-Symin)/actualHeight + Symin;
			float posY = (pixelY*(aabbMeters.w-aabbMeters.z))/float(actualHeight) + aabbMeters.z;

			float4 vertex;
			vertex.x = posX;
			vertex.y = posY;
			vertex.z = textureU;
			vertex.w = textureV;
			vertexBuffer[vertNum] = vertex;


			//Generate mesh
			// Quad configuration:
			// 0-1
			// |/|
			// 2-3
			// Index order: 0 - 1 - 2, 2 - 1 -3
			//Already loaded vertnum for 0
			int vertNum0 = 0;
			int vertNum1 = 0;
			int vertNum2 = 0;
			int vertNum3 = 0;

			//If degree greater than 0, assemble quad
			if(degree > 0)
			{
				//garunteed to be in range by nature of quadtree degree
				vertNum0 = vertNum;
				vertNum1 = quadTreeScanResults[(pixelX+degree) + (pixelY)*textureBufferSize];
				vertNum2 = quadTreeScanResults[(pixelX) + (pixelY+degree)*textureBufferSize];
				vertNum3 = quadTreeScanResults[(pixelX+degree) + (pixelY+degree)*textureBufferSize];
			}

			//Always fill buffer
			// Index order: 0 - 1 - 2, 2 - 1 -3
			int offset = vertNum*6;
			indexBuffer[offset+0] = vertNum0;
			indexBuffer[offset+1] = vertNum1;
			indexBuffer[offset+2] = vertNum2;
			indexBuffer[offset+3] = vertNum2;
			indexBuffer[offset+4] = vertNum1;
			indexBuffer[offset+5] = vertNum3;

		}
	}
}

__global__ void reshapeTextureKernel(int actualWidth, int actualHeight, int finalTextureWidth, int finalTextureHeight, int textureBufferSize, 
		Float4SOA planarTexture, float4* finalTexture)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int destIndex = x + y * finalTextureWidth;
	int sourceIndex = x + y * textureBufferSize;

	float4 textureValue = {CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F};

	if(x < actualWidth && y < actualHeight)
	{
		textureValue.x = planarTexture.x[sourceIndex];
		textureValue.y = planarTexture.y[sourceIndex];
		textureValue.z = planarTexture.z[sourceIndex];
		textureValue.w = planarTexture.w[sourceIndex];
	}

	
	finalTexture[destIndex] = textureValue;
}


__host__ void quadtreeMeshGeneration(glm::vec4 aabbMeters, int actualWidth, int actualHeight, int* quadTreeAssemblyBuffer,
									 int* quadTreeScanResults, int textureBufferSize, int* blockResults, int blockResultsBufferSize,
									 int* indexBuffer, float4* vertexBuffer, int* compactCount, int* host_compactCount, int outputBufferSize,
									 int finalTextureWidth, int finalTextureHeight, Float4SOA planarTexture, float4* finalTexture)
{
	int blockSize = roundupnextpow2(actualWidth);
	int numBlocks = actualHeight;
	dim3 threads(blockSize >> 1);//2 elements per thread
	dim3 blocks(numBlocks);
	int sharedCount = (blockSize+2)*sizeof(int);


	//Make sure size constraints aren't violated
	assert(blocks.x <= blockResultsBufferSize);
	assert(blockResultsBufferSize <= blockSize);

	//Scan blocks
	quadTreeExclusiveScanKernel<<<blocks,threads,sharedCount>>>(actualWidth, quadTreeAssemblyBuffer, 
		quadTreeScanResults, textureBufferSize, blockResults);

	//Scan block results
	int pow2 = roundupnextpow2(numBlocks);
	threads = dim3(pow2>>1);
	blocks = dim3(1);
	assert(pow2 <= blockResultsBufferSize);

	sharedCount = (pow2 + 2)*sizeof(int);
	blockResultsExclusiveScanKernel<<<blocks,threads,sharedCount>>>(blockResults, numBlocks, compactCount);

	cudaMemcpy(host_compactCount, compactCount, sizeof(int), cudaMemcpyDeviceToHost);

	//Reintegrate
	//Also scatter (generate meshes and vertecies in the process)
	threads = dim3(actualWidth);
	blocks = dim3(numBlocks);
	reintegrateResultsKernel<<<blocks,threads>>>(actualWidth, textureBufferSize, quadTreeScanResults, blockResults);

	assert(finalTextureWidth <= textureBufferSize);
	assert(finalTextureHeight <= textureBufferSize);

	scatterResultsKernel<<<blocks,threads>>>(aabbMeters, actualWidth, actualHeight, finalTextureWidth, finalTextureHeight, textureBufferSize, 
		quadTreeAssemblyBuffer, quadTreeScanResults, blockResults, indexBuffer, vertexBuffer);


	//Reshape texture to aligned memory
	threads = dim3(finalTextureWidth);
	blocks = dim3(finalTextureHeight);
	reshapeTextureKernel<<<blocks,threads>>>(actualWidth, actualHeight, finalTextureWidth, finalTextureHeight, textureBufferSize, 
		planarTexture, finalTexture);
}