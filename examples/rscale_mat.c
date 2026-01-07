#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("original:\n");
  mat_print(m);

  Mat *scaled = mat_rscale(m, 3);
  printf("mat_rscale(m, 3):\n");
  mat_print(scaled);

  printf("original (unchanged):\n");
  mat_print(m);

  mat_free_mat(m);
  mat_free_mat(scaled);
  return 0;
}
