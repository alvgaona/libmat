#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("original:\n");
  mat_print(m);

  Mat *copy = mat_copy(m);
  printf("copy:\n");
  mat_print(copy);

  mat_free_mat(m);
  mat_free_mat(copy);
  return 0;
}
