#include <stdio.h>
#include <math.h>

#include "common.h"

#define LENGTH 10
#define BLOCK_SIZE 1024

typedef float num_t;

void reset(num_t *xs, size_t length) {
  for (size_t i = 0; i < LENGTH; i++) {
    xs[i] = 0;
  }
}

void host_bscan(num_t *xs, num_t *ys, size_t length) {
  ys[0] = xs[0];
  for (size_t i = 1; i < length; i++) {
    ys[i] = ys[i - 1] + xs[i];
  }
}

__global__ 
void single_thread_bscan(num_t *xs, num_t *ys, size_t length) {
  int global_index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (global_index > 0) {
    return;
  }

  ys[0] = xs[0];
  for (size_t i = 1; i < length; i++) {
    ys[i] = ys[i - 1] + xs[i];
  }
}

void run_block_scan(int argc, char **argv) {
  size_t array_size = LENGTH * sizeof(num_t);

  num_t *xs = (num_t *)malloc(array_size);
  num_t *truth_ys = (num_t *)malloc(array_size);
  for (size_t i = 0; i < LENGTH; i++) {
    xs[i] = 1;
  }
  host_bscan(xs, truth_ys, LENGTH);

  num_t *device_xs = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_xs, array_size), "Couldn't allocate device_xs");
  CUDA_ERROR(
      cudaMemcpy(device_xs, xs, array_size, cudaMemcpyHostToDevice),
      "Couldn't copy to device_xs");
  num_t *device_ys = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_ys, array_size), "Couldn't allocate device_ys");

  int num_blocks = ceil(float(LENGTH) / float(BLOCK_SIZE));
  printf("%d\n", array_size);
  single_thread_bscan<<<num_blocks, BLOCK_SIZE>>>(device_xs, device_ys, LENGTH);

  cudaDeviceSynchronize();

  num_t *ys = (num_t *)malloc(array_size);
  CUDA_ERROR(
      cudaMemcpy(ys, device_ys, array_size, cudaMemcpyDeviceToHost),
      "Couldn't copy to ys");

  for (size_t i = 0; i < LENGTH; i++) {
    if (truth_ys[i] != ys[i]) {
      fprintf(
          stderr,
          "Incorrect value at index %d: Expected %f, got %f\n",
          i,
          truth_ys[i],
          ys[i]);
    }
  }

  printf("Result: ");
  for (size_t i = 0; i < LENGTH; i++) {
    printf("%f, ", ys[i]);
  }
  printf("\n");
}

