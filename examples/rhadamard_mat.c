#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals1[] = {1, 2, 3, 4};
  mat_elem_t vals2[] = {5, 6, 7, 8};
  Mat *m1 = mat_from(2, 2, vals1);
  Mat *m2 = mat_from(2, 2, vals2);

  printf("m1:\n");
  mat_print(m1);
  printf("m2:\n");
  mat_print(m2);

  Mat *result = mat_rhadamard(m1, m2);
  printf("mat_rhadamard(m1, m2):\n");
  mat_print(result);

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(result);
  return 0;
}
