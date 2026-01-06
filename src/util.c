#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "mat.h"

void print_mat(Mat *mat) {
  assert(mat != NULL);
  assert(mat->data != NULL);

  printf("[\n");
  for (size_t i = 0; i < mat->rows; i++) {
    printf(" ");
    for (size_t j = 0; j < mat->cols; j++) {
      printf("%g", mat->data[i * mat->cols + j]);
      if (j < mat->cols - 1) {
        printf(" ");
      }
    }
    if (i < mat->rows - 1) {
      printf(";");
    }
    printf("\n");
  }
  printf("]\n");
}

size_t *flatten_mat(size_t *in, size_t rows, size_t cols) {
  assert(in != NULL);
  assert(rows > 0);
  assert(cols > 0);
  
  size_t *out = calloc(rows * cols, sizeof(size_t));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out[i * cols + j] = in[i * cols + j];
    }
  }

  return out;
}
