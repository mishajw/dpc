#define CUDA_ERROR(statement, message) \
  do { \
    cudaError_t err = statement; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err)); \
    } \
  } while (0);
