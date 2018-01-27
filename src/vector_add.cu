#include <stdio.h>
#include <math.h>

#define CUDA_ERROR(statement, message) \
  do { \
    err = statement; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err)); \
    } \
  } while (0);

__global__ void add_array(int *a, int *b, int *result, int size) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < size) {
    result[index] = a[index] + b[index];
  }
}

void run_vector_add(int argc, char **argv) {
  cudaError_t err;
  int num_elements = 1024;
  size_t size = num_elements * sizeof(int);
  int *a = (int *) malloc(size);
  int *b = (int *) malloc(size);
  for (int i = 0; i < num_elements; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  int *device_a = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_a, size), "Failed to allocate device_a");
  CUDA_ERROR(cudaMemcpy(device_a, a, size, cudaMemcpyHostToDevice), "Failed to copy to device_a");

  int *device_b = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_b, size), "Failed to allocate device_b");
  CUDA_ERROR(cudaMemcpy(device_b, b, size, cudaMemcpyHostToDevice), "Failed to copy to device_a");

  int *device_result = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_result, size), "Failed to allocate device_result");
  
  int num_threads_per_block = 32;
  int num_blocks = ceil(num_elements / num_threads_per_block);
  add_array<<<num_blocks, num_threads_per_block>>>(
      device_a, device_b, device_result, num_elements);

  cudaDeviceSynchronize();
  int *result = (int *) malloc(size);
  CUDA_ERROR(
      cudaMemcpy(result, device_result, size, cudaMemcpyDeviceToHost),
      "Failed to copy from device_result");

  for (int i = 0; i < num_elements; i++) {
    printf("%d, %d, %d\n", a[i], b[i], result[i]);
  }

  free(a);
  free(b);
  free(result);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_result);
}

