#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3};
  Mat *m = mat_diag_from(3, vals);

  printf("diagonal matrix:\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
