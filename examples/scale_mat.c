#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("original:\n");
  mat_print(m);

  mat_scale(m, 2.5);
  printf("after mat_scale(m, 2.5):\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
