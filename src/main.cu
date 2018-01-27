#include <stdio.h>

#include "vector_add.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: dpc <program> ...\n");
    exit(1);
  }

  int new_argc = argc - 2;
  char **new_argv = argv + 2;

  char *program = argv[1];
  if (strcmp(program, "vector_add") == 0) {
    run_vector_add(new_argc, new_argv);
  }
}

