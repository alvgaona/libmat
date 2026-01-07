#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals1[] = {1, 2, 3};
  mat_elem_t vals2[] = {4, 5, 6};

  Vec *v1 = mat_vec_from(3, vals1);
  Vec *v2 = mat_vec_from(3, vals2);

  printf("v1:\n");
  mat_print(v1);
  printf("v2:\n");
  mat_print(v2);

  mat_elem_t dot = mat_dot(v1, v2);
  printf("mat_dot(v1, v2) = %g\n", dot);

  mat_free_mat(v1);
  mat_free_mat(v2);
  return 0;
}
