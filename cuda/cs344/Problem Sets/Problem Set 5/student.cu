/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <stdio.h>
#include "utils.h"

#define NUMBINS 1024

__global__
void yourHisto1(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
   for (int i = 0; i < numVals; ++i)
     histo[vals[i]]++;
    
}
__global__
void yourHisto2(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < numVals) {
        __shared__ int localBin[NUMBINS];
        if (threadIdx.x < NUMBINS) {
            localBin[threadIdx.x] = 0;
        }
        __syncthreads();

        atomicAdd(&localBin[vals[index]],1);     
        __syncthreads();
        if (threadIdx.x < NUMBINS) {
            atomicAdd(&histo[threadIdx.x], localBin[threadIdx.x]);
        }       
   }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
  //yourHisto1<<<1,1>>>(d_vals,d_histo, numElems);
  int numBlocks(numElems/1024 + 1), numThreads(1024);
  yourHisto2<<<numBlocks,numThreads>>>(d_vals,d_histo, numElems);
  printf("numBins %d numElems %d\n", numBins, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
