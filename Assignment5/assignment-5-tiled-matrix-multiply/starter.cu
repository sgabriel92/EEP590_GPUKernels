
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

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float ds_A[16][16];
  __shared__ float ds_B[16][16];
  int bx = hipBlockIdx_x;
  int by = hipBlockIdx_y;
  int tx = hipThreadIdx_x;
  int ty = hipThreadIdx_y;
  int Row = by * 16 + ty;
  int Col = bx * 16 + tx;
  float Pvalue = 0;
  for (int m = 0; m < (numAColumns - 1) / 16 + 1; ++m) {
    if (Row < numARows && m * 16 + tx < numAColumns)
      ds_A[ty][tx] = A[Row * numAColumns + m * 16 + tx];
    else
      ds_A[ty][tx] = 0;
    if (Col < numBColumns && m * 16 + ty < numBRows)
      ds_B[ty][tx] = B[(m * 16 + ty) * numBColumns + Col];
    else
      ds_B[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < 16; ++k)
      Pvalue += ds_A[ty][k] * ds_B[k][tx];
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns)
    C[Row * numCColumns + Col] = Pvalue;
  
 
    
  

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(hipMalloc(&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(hipMalloc(&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(hipMalloc(&deviceC, numCRows * numCColumns * sizeof(float)));


  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(hipMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), hipMemcpyHostToDevice));
  wbCheck(hipMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), hipMemcpyHostToDevice));


  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(16, 16, 1);
  dim3 DimGrid((numCColumns + DimBlock.x - 1) / DimBlock.x, (numCRows + DimBlock.y - 1) / DimBlock.y, 1);

 


  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
 hipLaunchKernelGGL(matrixMultiplyShared, DimGrid, DimBlock, 0, 0, deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);


  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(hipMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), hipMemcpyDeviceToHost));


  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(hipFree(deviceA));
  wbCheck(hipFree(deviceB));
  wbCheck(hipFree(deviceC));


  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
