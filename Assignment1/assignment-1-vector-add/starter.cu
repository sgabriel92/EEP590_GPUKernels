#include <hip/hip_runtime.h>
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len) {
     out[i] = in1[i] + in2[i];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t Nbytes = inputLength * sizeof(float);
  
  wbCheck(hipMalloc(&deviceInput1,Nbytes));
  wbCheck(hipMalloc(&deviceInput2,Nbytes));
  wbCheck(hipMalloc(&deviceOutput,Nbytes));


  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(hipMemcpy(deviceInput1,hostInput1,Nbytes,hipMemcpyHostToDevice));
  wbCheck(hipMemcpy(deviceInput2,hostInput2,Nbytes,hipMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid = dim3((inputLength+256-1)/256,1,1); 
  dim3 DimBlock = dim3(256, 1, 1);

  wbTime_start(Compute, "Performing HIP computation");

  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(vecAdd, DimGrid, DimBlock, 0, 0, deviceInput1, deviceInput2, deviceOutput, inputLength);
  hipDeviceSynchronize();
  wbCheck(hipGetLastError());

  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(hipMemcpy(hostOutput,deviceOutput,Nbytes,hipMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(hipFree(deviceInput1));
  wbCheck(hipFree(deviceInput2));
  wbCheck(hipFree(deviceOutput));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
