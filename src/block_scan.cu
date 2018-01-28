#include <stdio.h>
#include <math.h>

#include "common.h"

#define LENGTH 10
#define BLOCK_SIZE 1024

typedef float num_t;

const size_t ARRAY_SIZE = (LENGTH * sizeof(num_t));

void reset(num_t *input, size_t length) {
  for (size_t i = 0; i < LENGTH; i++) {
    input[i] = 0;
  }
}

void host_bscan(num_t *input, num_t *result, size_t length) {
  result[0] = input[0];
  for (size_t i = 1; i < length; i++) {
    result[i] = result[i - 1] + input[i];
  }
}

__global__ 
void single_thread_bscan(num_t *input, num_t *result, size_t length) {
  int index = GLOBAL_INDEX;

  if (index > 0) {
    return;
  }

  result[0] = input[0];
  for (size_t i = 1; i < length; i++) {
    result[i] = result[i - 1] + input[i];
  }
}

__global__
void hsh_nsm_bscan(num_t *input, num_t *result, size_t length) {
  // TODO: This should not use shared memory, but in order to solve the read/write conflict
  // between blocks, we need to create an array 2x the size of `length` - how to solve this
  // easily without shared memory?

  int index = GLOBAL_INDEX;

  for (int stride = 1; stride <= length / 2; stride *= 2) {
    __syncthreads();

    bool should_add = index >= stride && index < length;

    if (!should_add) {
      continue;
    }

    input[index] = input[index] + input[index - stride];
  }

  result[index] = input[index];
}

__global__
void hsh_bscan(num_t *input, num_t *result, size_t length) {
  __shared__ num_t buffer[BLOCK_SIZE * 2];
  num_t *read_buffer = buffer;
  num_t *write_buffer = &buffer[blockDim.x];

  int index = GLOBAL_INDEX;

  if (index < length) {
    read_buffer[threadIdx.x] = input[index];
  }

  for (int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
    __syncthreads();

    if (threadIdx.x >= stride) {
      write_buffer[threadIdx.x] = read_buffer[threadIdx.x] + read_buffer[threadIdx.x - stride];
    } else {
      write_buffer[threadIdx.x] = read_buffer[threadIdx.x];
    }

    // Switch the read and write buffers
    num_t *tmp_buffer = read_buffer;
    read_buffer = write_buffer;
    write_buffer = tmp_buffer;
  }

  if (index < length) {
    result[index] = read_buffer[threadIdx.x];
  }
}

void test_function(void (*func)(num_t*, num_t*, size_t), num_t *input, num_t *truth) {
  // Set up device arrays
  num_t *device_input = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_input, ARRAY_SIZE), "Couldn't allocate device_input");
  CUDA_ERROR(
      cudaMemcpy(device_input, input, ARRAY_SIZE, cudaMemcpyHostToDevice),
      "Couldn't copy to device_input");
  num_t *device_result = NULL;
  CUDA_ERROR(cudaMalloc((void **) &device_result, ARRAY_SIZE), "Couldn't allocate device_result");

  // Run the function to test
  int num_blocks = ceil(float(LENGTH) / float(BLOCK_SIZE));
  func<<<num_blocks, BLOCK_SIZE>>>(device_input, device_result, LENGTH);

  // Wait for the function to finish
  cudaDeviceSynchronize();

  // Copy the results into host memory
  num_t *result = (num_t *)malloc(ARRAY_SIZE);
  CUDA_ERROR(
      cudaMemcpy(result, device_result, ARRAY_SIZE, cudaMemcpyDeviceToHost),
      "Couldn't copy to result");

  // Check it against the truth
  size_t num_incorrect = 0;
  for (size_t i = 0; i < LENGTH; i++) {
    if (truth[i] != result[i]) {
      fprintf(
          stderr,
          "Incorrect value at index %d: Expected %f, got %f\n",
          i,
          truth[i],
          result[i]);
      num_incorrect++;
    }
  }
  printf("Number of incorrect results: %ld\n", num_incorrect);
}

void run_block_scan(int argc, char **argv) {
  num_t *input = (num_t *)malloc(ARRAY_SIZE);
  num_t *truth = (num_t *)malloc(ARRAY_SIZE);
  for (size_t i = 0; i < LENGTH; i++) {
    input[i] = 1;
  }
  host_bscan(input, truth, LENGTH);

  test_function(single_thread_bscan, input, truth);
}

