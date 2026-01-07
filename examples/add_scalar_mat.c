#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("original:\n");
  mat_print(m);

  mat_add_scalar(m, 10);
  printf("after mat_add_scalar(m, 10):\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
