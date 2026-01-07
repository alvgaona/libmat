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

  printf("m1 (2x3):\n");
  mat_print(m1);
  printf("m2 (3x2):\n");
  mat_print(m2);

  Mat *result = mat_rmul(m1, m2);
  printf("m1 * m2 (2x2):\n");
  mat_print(result);

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(result);
  return 0;
}
