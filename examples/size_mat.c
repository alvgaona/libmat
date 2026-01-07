#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  Mat *m = mat_ones(3, 4);

  MatSize size = mat_size(m);
  printf("rows: %zu, cols: %zu\n", size.x, size.y);

  mat_free_mat(m);
  return 0;
}
