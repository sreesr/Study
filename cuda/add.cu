#include <stdio.h>
#include <errno.h>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    printf("%d %d %d %d\n", bx, by, tx, ty);
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
  
}

int main(void)
{
  int N = 1<<20;
  float *x = NULL, *y = NULL;

  // Allocate Unified Memory – accessible from CPU or GPU
  int rc = cudaMalloc(&x, N*sizeof(float)) ;
  if (cudaSuccess != rc) {
    printf("failed %s\n", cudaGetErrorString(cudaGetLastError()));
    exit(1);
  }
  //rc = cudaMallocManaged(&y, N*sizeof(float));
  rc = cudaMalloc(&y, N*sizeof(float)) ;
  if (cudaSuccess != rc) {
    printf("failed %s\n", cudaGetErrorString(cudaGetLastError()));
    exit(1);
  }

  if (x == NULL)  {
    printf("x == NULL\n");
    exit (1);
  }

  float *h_x = (float *) malloc(sizeof(float)*N);
  float *h_y = (float *) malloc(sizeof(float)*N);
  

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    h_y[i] = 2.0f;
    h_x[i] = 1.0f;
  }

  // copy host memory to device
  cudaError_t error = cudaMemcpy(x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
	  printf("cudaMemcpy (d_A,h_x) returned error %s %d %d" , cudaGetErrorString(error) , error , __LINE__);
	  exit(1);
  }
  printf("Hi there\n");
  //exit(1);

  
  // Run kernel on 1M elements on the GPU
  add<<<1, 256>>>(N, x, y);
  printf("%d\n", __LINE__);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //printf("%d\n", __LINE__);
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  // copy host memory to device
  error = cudaMemcpy(h_y, y, sizeof(h_x), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
	  printf("cudaMemcpy (d_A,h_x) returned error %s %d %d" , cudaGetErrorString(error) , error , __LINE__);
	  exit(1);
  }  
  for (int j = 0; j < N; j++)
    maxError = fmax(maxError, fabs(h_y[j]-3.0f));
  printf("Max error: %f\n" , maxError);

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}

