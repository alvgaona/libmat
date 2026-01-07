#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  Mat *m = mat_ones(2, 3);

  printf("2x3 ones:\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
