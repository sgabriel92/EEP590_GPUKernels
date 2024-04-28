// Histogram Equalization
#include<hip/hip_runtime.h>
#include <wb.h>

#define HISTOGRAM_LENGTH 256

//Helper
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ insert code here
//Implementation of Kernels
//Step 1
//Cast Float to UnsignedChar
__global__ void castFloatToUnsignedChar(unsigned char *output, float *input, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < width && y < height) {
    int idx = y * width + x;
    output[channels * idx] = (unsigned char)(255 * input[channels * idx]);     // r
    output[channels * idx + 1] = (unsigned char)(255 * input[channels * idx + 1]); // g
    output[channels * idx + 2] = (unsigned char)(255 * input[channels * idx + 2]); // b
    //printf("Value %d: %u\n", idx, output[channels * idx]);
  }
}

//Step 2
// Convert RGB to Grayscale
__global__ void rgbToGrayscale(unsigned char *grayImage, unsigned char *rgbImage, int width, int height, int channels) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (col < width && row < height) {
    int idx = row * width + col;
    unsigned char r = rgbImage[channels * idx];
    unsigned char g = rgbImage[channels * idx + 1];
    unsigned char b = rgbImage[channels * idx + 2];
    grayImage[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

//Step 3
//Compute Histogram
__global__ void computeHistogram(unsigned int *histogram, unsigned char *grayImage, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    atomicAdd(&(histogram[grayImage[i]]), 1);
  }
}

//Step 4
//Compute CDF
__global__ void computeCDF(float *cdf, unsigned int *histogram, int size) {
  __shared__ unsigned int scan_array[HISTOGRAM_LENGTH];
  
  int idx = threadIdx.x;
  if (idx < size) {
    scan_array[idx] = histogram[idx];
    __syncthreads();

    for (int stride = 1; stride <= idx; stride *= 2) {
      unsigned int in = 0;
      if (idx >= stride) {
        in = scan_array[idx - stride];
      }
      __syncthreads();
      scan_array[idx] += in;
      __syncthreads();
    }

    cdf[idx] = scan_array[idx];
  }
}

//Step 6
//Histogram Equalization
__device__ float clamp(float x, float start, float end) {
    //printf("Check1");
    return min(max(x, start), end);
}

__device__ unsigned char correct_color(float val, float* cdf, float cdf_min) {
    //printf("Check2");
    return clamp((255 * (cdf[(int)(val * 255)] - cdf_min)) / (1.0 - cdf_min), 0, 255);
}

__global__ void applyHistogramEqualization(unsigned char *ucharImageOut, unsigned char *ucharImage, float *cdf, float cdfmin, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int i = y * width + x;
    ucharImageOut[i * 3] = correct_color(ucharImage[i * 3] / 255.0f, cdf, cdfmin);     // r
    ucharImageOut[i * 3 + 1] = correct_color(ucharImage[i * 3 + 1] / 255.0f, cdf, cdfmin); // g
    ucharImageOut[i * 3 + 2] = correct_color(ucharImage[i * 3 + 2] / 255.0f, cdf, cdfmin); // b
  }
}

//Step 7
//Cast unsigned char back to float
__global__ void castUnsignedCharToFloat(float *output, unsigned char *input, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < width && y < height) {
    int idx = y * width + x;
    output[channels * idx] = (float)(input[channels * idx]) / 255.0f;     // r
    output[channels * idx + 1] = (float)(input[channels * idx + 1]) / 255.0f; // g
    output[channels * idx + 2] = (float)(input[channels * idx + 2]) / 255.0f; // b
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData; 
  unsigned char *deviceIntermediateImageData; 
  unsigned char *deviceIntermediateImageDataTWO; 
  unsigned char *deviceIntermediateGrayscale;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  float *deviceCDFnormalized;
  float *hostCDF;
  float *normalizedCDF;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage); //modified
  hostOutputImageData = wbImage_getData(outputImage);//modified
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  //@@ insert code here
  //CrossCheck Information
  printf("Image dimensions: %d x %d x %d\n", imageWidth, imageHeight, imageChannels);
  printf("Image size: %d\n", imageWidth * imageHeight * imageChannels);
  printf("hostInputImageData: %p\n", hostInputImageData);
  printf("hostOutputImageData: %p\n", hostOutputImageData);
  //

  //Allocate GPU memory:
  wbTime_start(GPU, "Allocating GPU memory.");
  int imageSize = imageWidth * imageHeight * imageChannels;
  int imageSizeGray = imageWidth * imageHeight;
  size_t size = imageSize * sizeof(float);
  size_t size_uc = imageSize * sizeof(unsigned char);
  size_t size_uc_gray = imageSizeGray * sizeof(unsigned char);
  size_t size_histo = HISTOGRAM_LENGTH * sizeof(unsigned int);
  size_t size_cdf = HISTOGRAM_LENGTH * sizeof(float);

  wbCheck(hipMalloc((void **)&deviceInputImageData, size));
  wbCheck(hipMalloc((void **)&deviceIntermediateImageData, size_uc));
  wbCheck(hipMalloc((void **)&deviceIntermediateImageDataTWO, size_uc));
  wbCheck(hipMalloc((void **)&deviceIntermediateGrayscale, size_uc_gray));
  wbCheck(hipMalloc((void **)&deviceHistogram,size_histo));
  hipMemset(deviceHistogram, 0, size_histo);
  wbCheck(hipMalloc((void **)&deviceCDF, size_cdf));
  wbCheck(hipHostMalloc((void **)&hostCDF, size_cdf));
  wbCheck(hipHostMalloc((void **)&normalizedCDF, size_cdf));
  wbCheck(hipMalloc((void **)&deviceCDFnormalized, size_cdf));
  wbCheck(hipMalloc((void **)&deviceOutputImageData, size));
  wbTime_stop(GPU, "Allocating GPU memory.");

  //Copy to GPU
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(hipMemcpy(deviceInputImageData, hostInputImageData, size, hipMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //Initiate grid and block dimension for image casting
  // Define dimensions for image processing
  int blockWidth = 16;
  dim3 dimBlockImage(blockWidth, blockWidth, 1);
  dim3 dimGridImage((imageWidth + blockWidth - 1) / blockWidth, (imageHeight + blockWidth - 1) / blockWidth, 1);

  //Define grid and block dimensions for histogram and cdf kernels
  int blockSize = 256;
  dim3 DimBlockHisto(blockSize, 1, 1);
  dim3 DimGridHisto((imageSizeGray + blockSize - 1) / blockSize, 1, 1); 



  // Process
  wbTime_start(Compute, "Performing HIP computation");

  //Step 1
  hipLaunchKernelGGL(castFloatToUnsignedChar, dimGridImage, dimBlockImage, 0, 0, deviceIntermediateImageData, deviceInputImageData, imageWidth, imageHeight,imageChannels);
  hipDeviceSynchronize();
  printf("Step 1 done\n");
  printf("Debug Information:\n");
  printf("First element of deviceInputImageData: %f\n", deviceInputImageData[0]);
  printf("First element of deviceIntermediateImageData: %u\n", deviceIntermediateImageData[0]);
  //Print Image Info
  // for (int i = 4950; i < 5000; i++) {
  //     printf("Value %d: %f\n", i, deviceInputImageData[i]);
  // }


  //Step 2
  hipLaunchKernelGGL(rgbToGrayscale, dimGridImage, dimBlockImage, 0, 0, deviceIntermediateGrayscale, deviceIntermediateImageData, imageWidth, imageHeight, imageChannels);
  hipDeviceSynchronize();
  printf("Step 2 done\n");
  printf("Debug Information:\n");
  printf("First element of deviceIntermediateGrayscale: %u\n", deviceIntermediateGrayscale[0]);
  printf("Last element of deviceIntermediateGrayscale: %u\n", deviceIntermediateGrayscale[65536-1]);
  //Print Grayscale Image
  // for (int i = 1900; i < 2000; i++) {
  //     printf("Value %d: %u\n", i, deviceIntermediateGrayscale[i]);
  // }

  //Step 3
  hipLaunchKernelGGL(computeHistogram, DimGridHisto, DimBlockHisto, 0, 0, deviceHistogram, deviceIntermediateGrayscale, imageSizeGray);
  hipDeviceSynchronize();
  printf("Step 3 done\n");
  printf("Debug Information:\n");
  printf("Last element of deviceHistogram: %u\n", deviceHistogram[255]);
  //Print the histogram
  // for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
  //     printf("Bin %d: %u\n", i, deviceHistogram[i]);
  // }

  //Step 4
  hipLaunchKernelGGL(computeCDF, DimGridHisto, DimBlockHisto, 0, 0, deviceCDF, deviceHistogram, HISTOGRAM_LENGTH);
  hipDeviceSynchronize();
  printf("Step 4 done\n");
  printf("Debug Information:\n");
  printf("First element of deviceCDF: %f\n", deviceCDF[0]);
  printf("Last element of deviceCDF: %f\n", deviceCDF[255]);
  //Print the cdf
  // for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
  //     printf("Value %d: %f\n", i, deviceCDF[i]);
  // }

  // Step 5
  //Compute Minimum and Max Value of CDF
  wbCheck(hipMemcpy(hostCDF, deviceCDF, size_histo, hipMemcpyDeviceToHost));
  printf("First element of hostCDF: %f\n", hostCDF[0]);
  printf("Last element of hostCDF: %f\n", hostCDF[255]);
  //Print the histogram before adjustment
  // for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
  //     printf("Value %d: %f\n", i, hostCDF[i]);
  // }

  float total = hostCDF[HISTOGRAM_LENGTH - 1];
  //float normalizedCDF[HISTOGRAM_LENGTH];
  for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
      normalizedCDF[i] = hostCDF[i] / (float)total;
  }

  // Print the normalized CDF 
  // for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
  //     printf("Value %d: %f\n", i, normalizedCDF[i]);
  // }

  float max_cdf = normalizedCDF[HISTOGRAM_LENGTH - 1]; 
  float min_cdf = normalizedCDF[0];
  wbCheck(hipMemcpy(deviceCDFnormalized, normalizedCDF, size_cdf, hipMemcpyHostToDevice));

  printf("min_cdf: %f, max_cdf: %f\n", min_cdf, max_cdf);
  printf("Step 5 done\n");
  printf("Debug Information:\n");
  printf("First element of hostCDF: %f\n", hostCDF[0]);
  printf("Last element of hostCDF: %f\n", hostCDF[255]);
  printf("First element of normalizedCDF: %f\n", normalizedCDF[0]);
  printf("Last element of normalizedCDF: %f\n", normalizedCDF[255]);

  // Step 6
  //Apply color correction to unchar image
  hipLaunchKernelGGL(applyHistogramEqualization, dimGridImage, dimBlockImage, 0, 0, deviceIntermediateImageDataTWO, deviceIntermediateImageData, deviceCDFnormalized, min_cdf, imageWidth, imageHeight);
  hipDeviceSynchronize();
  printf("Step 6 done\n");
  printf("Debug Information:\n");
  // Print intermediate data
  // for (int i = 4950; i < 5000; i++) {
  //     printf("Value %d: %u\n", i, deviceIntermediateImageDataTWO[i]);
  // }

  //Step 7 
  //Cast back to float
  hipLaunchKernelGGL(castUnsignedCharToFloat, dimGridImage, dimBlockImage, 0, 0, deviceOutputImageData, deviceIntermediateImageDataTWO, imageWidth, imageHeight, imageChannels);
  hipDeviceSynchronize();
  printf("Step 7 done\n");
  printf("Debug Information:\n");

  // Copy data back to host
  wbCheck(hipMemcpy(hostOutputImageData, deviceOutputImageData, size, hipMemcpyDeviceToHost));
  //print output on host
  // for (int i = 4950; i < 5000; i++) {
  //       printf("Value %d: %f\n", i, hostOutputImageData[i]);
  //   }
  wbTime_stop(Compute, "Performing HIP computation");

 
  //@@ insert code here
  wbTime_start(GPU, "Freeing GPU Memory");
  // Free GPU memory
  wbCheck(hipFree(deviceInputImageData));
  wbCheck(hipFree(deviceIntermediateImageData));
  wbCheck(hipFree(deviceIntermediateImageDataTWO));
  wbCheck(hipFree(deviceIntermediateGrayscale));
  wbCheck(hipFree(deviceHistogram));
  wbCheck(hipFree(deviceCDF));
  wbCheck(hipFree(deviceCDFnormalized));
  wbCheck(hipFree(deviceOutputImageData));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, outputImage);

  //@@ insert code here

  free(inputImage);
  free(outputImage);
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
