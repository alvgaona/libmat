#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals1[] = {1, 2, 3, 4};
  mat_elem_t vals2[] = {10, 20, 30, 40};
  mat_elem_t vals3[] = {100, 200, 300, 400};

  Mat *m1 = mat_from(2, 2, vals1);
  Mat *m2 = mat_from(2, 2, vals2);
  Mat *m3 = mat_from(2, 2, vals3);

  printf("m1:\n");
  mat_print(m1);
  printf("m2:\n");
  mat_print(m2);
  printf("m3:\n");
  mat_print(m3);

  Mat *result = mat_radd_many(3, m1, m2, m3);
  printf("mat_radd_many(3, m1, m2, m3):\n");
  mat_print(result);

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(m3);
  mat_free_mat(result);
  return 0;
}
