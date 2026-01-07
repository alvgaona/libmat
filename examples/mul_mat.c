#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  // 2x3 matrix
  mat_elem_t vals1[] = {1, 2, 3, 4, 5, 6};
  Mat *m1 = mat_from(2, 3, vals1);

  // 3x2 matrix
  mat_elem_t vals2[] = {7, 8, 9, 10, 11, 12};
  Mat *m2 = mat_from(3, 2, vals2);

  // Output 2x2 matrix
  Mat *out = mat_mat(2, 2);

  printf("m1 (2x3):\n");
  mat_print(m1);
  printf("m2 (3x2):\n");
  mat_print(m2);

  mat_mul(out, m1, m2);
  printf("mat_mul(out, m1, m2):\n");
  mat_print(out);

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(out);
  return 0;
}
