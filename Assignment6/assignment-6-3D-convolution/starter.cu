#include <wb.h>
#include <hip/hip_runtime.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "HIP error: ", hipGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define MASK_WIDTH  3
#define MASK_RADIUS MASK_WIDTH/2

__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < x_size && y < y_size && z < z_size) {
    float Pvalue = 0.0;

    for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
      for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
        for (int k = -MASK_RADIUS; k <= MASK_RADIUS; k++) {
          if (z + k >= 0 && z + k < z_size &&
              y + j >= 0 && y + j < y_size &&
              x + i >= 0 && x + i < x_size) {
            Pvalue += input[(z + k) * y_size * x_size + (y + j) * x_size + (x + i)] * M[k + MASK_RADIUS][j + MASK_RADIUS][i + MASK_RADIUS];
          }
        }
      }
    }

    output[z * y_size * x_size + y * x_size + x] = Pvalue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  wbCheck(hipMalloc((void **)&deviceInput, inputLength * sizeof(float)));
  wbCheck(hipMalloc((void **)&deviceOutput, inputLength * sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  wbCheck(hipMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), hipMemcpyHostToDevice));
  wbCheck(hipMemcpyToSymbol(M, hostKernel, kernelLength * sizeof(float)));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 dimBlock(8, 8, 8);
  dim3 dimGrid((x_size + dimBlock.x - 1) / dimBlock.x, 
              (y_size + dimBlock.y - 1) / dimBlock.y, 
              (z_size + dimBlock.z - 1) / dimBlock.z);
  hipLaunchKernelGGL(conv3d, dimGrid, dimBlock, 0, 0, deviceInput, deviceOutput, z_size, y_size, x_size);
  hipDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  wbCheck(hipMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), hipMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  hipFree(deviceInput);
  hipFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}