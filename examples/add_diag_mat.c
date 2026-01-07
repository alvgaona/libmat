#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t mat_vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, mat_vals);

  mat_elem_t diag_vals[] = {10, 20};
  Mat *d = mat_diag_from(2, diag_vals);

  printf("m:\n");
  mat_print(m);
  printf("diagonal matrix:\n");
  mat_print(d);

  Mat *result = mat_radd(m, d);

  printf("m + diag:\n");
  mat_print(result);

  mat_free_mat(m);
  mat_free_mat(d);
  mat_free_mat(result);
  return 0;
}
