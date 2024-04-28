// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <hip/hip_runtime.h>
#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  __shared__ volatile int partialSum[2*BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(start + tid < len)
  {
    partialSum[tid] = input[start + tid];
  }
  else
  {
    partialSum[tid] = 0.0;
  }
  if(start + blockDim.x + tid < len)
  {
    partialSum[blockDim.x + tid] = input[start + blockDim.x + tid];
  }
  else
  {
    partialSum[blockDim.x + tid] = 0.0;
  }
  __syncthreads();

  for(unsigned int stride = blockDim.x; stride > 64; stride >>= 1)
  {
    if(tid < stride)
    {
      partialSum[tid] += partialSum[tid + stride];
    }
    __syncthreads();
  }

  if(tid < 64)
  {

    partialSum[tid] += partialSum[tid + 64];
    partialSum[tid] += partialSum[tid + 32];
    partialSum[tid] += partialSum[tid + 16];
    partialSum[tid] += partialSum[tid + 8];
    partialSum[tid] += partialSum[tid + 4];
    partialSum[tid] += partialSum[tid + 2];
    partialSum[tid] += partialSum[tid + 1];
  }

  if(tid == 0)
  {
    output[blockIdx.x] = partialSum[0];
  }

}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t size = numInputElements * sizeof(float);
  size_t sizeOut = numOutputElements * sizeof(float); 
  
  wbCheck(hipMalloc(&deviceInput, size));
  wbCheck(hipMalloc(&deviceOutput, sizeOut));


  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(hipMemcpy(deviceInput, hostInput,size, hipMemcpyHostToDevice));


  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid((numInputElements+(float)BLOCK_SIZE-1/(float)BLOCK_SIZE), 1, 1);

  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(total,DimGrid,DimBlock,0,0,deviceInput,deviceOutput,numInputElements);

  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(hipMemcpy(hostOutput, deviceOutput, sizeOut, hipMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(hipFree(deviceInput));
  wbCheck(hipFree(deviceOutput));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
