#include "CudaUtils.h"


void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error %d: %s: %s.\n", err, msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 
