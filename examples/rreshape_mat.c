#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("m (2x2):\n");
  mat_print(m);

  Mat *reshaped = mat_rreshape(m, 4, 1);

  printf("reshaped (4x1):\n");
  mat_print(reshaped);

  mat_free_mat(m);
  mat_free_mat(reshaped);
  return 0;
}
