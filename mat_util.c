#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "mat.h"

void mat_print(Mat *mat) {
  assert(mat != NULL);
  assert(mat->data != NULL);

  printf("[");
  for (size_t i = 0; i < mat->rows; i++) {
    if (i > 0) printf(" ");
    for (size_t j = 0; j < mat->cols; j++) {
      printf("%g", mat->data[i * mat->cols + j]);
      if (j < mat->cols - 1) {
        printf(" ");
      }
    }
    if (i < mat->rows - 1) {
      printf(";\n");
    }
  }
  printf("]\n");
}

