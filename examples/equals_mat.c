#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals1[] = {1, 2, 3, 4};
  mat_elem_t vals2[] = {1, 2, 3, 4};
  mat_elem_t vals3[] = {1, 2, 3, 5};

  Mat *m1 = mat_from(2, 2, vals1);
  Mat *m2 = mat_from(2, 2, vals2);
  Mat *m3 = mat_from(2, 2, vals3);

  printf("m1:\n");
  mat_print(m1);
  printf("m2:\n");
  mat_print(m2);
  printf("m3:\n");
  mat_print(m3);

  printf("m1 == m2: %s\n", mat_equals(m1, m2) ? "true" : "false");
  printf("m1 == m3: %s\n", mat_equals(m1, m3) ? "true" : "false");

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(m3);
  return 0;
}
