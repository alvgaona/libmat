#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals1[] = {1, 2, 3, 4};
  mat_elem_t vals2[] = {5, 6, 7, 8};
  Mat *m1 = mat_from(2, 2, vals1);
  Mat *m2 = mat_from(2, 2, vals2);
  Mat *out = mat_mat(2, 2);

  printf("m1:\n");
  mat_print(m1);
  printf("m2:\n");
  mat_print(m2);

  mat_hadamard(out, m1, m2);
  printf("mat_hadamard(out, m1, m2):\n");
  mat_print(out);

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(out);
  return 0;
}
