#include <stdio.h>

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

void run_block_scan(int argc, char **argv) {
  num_t *xs = (num_t *)malloc(LENGTH * sizeof(num_t));
  num_t *ys = (num_t *)malloc(LENGTH * sizeof(num_t));
  for (size_t i = 0; i < LENGTH; i++) {
    xs[i] = 1;
  }

  host_bscan(xs, ys, LENGTH);

  printf("Result: ");
  for (size_t i = 0; i < LENGTH; i++) {
    printf("%f, ", ys[i]);
  }
  printf("\n");
}

