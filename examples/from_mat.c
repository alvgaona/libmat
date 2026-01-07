#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4, 5, 6};
  Mat *m = mat_from(2, 3, vals);

  printf("2x3 from values:\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
