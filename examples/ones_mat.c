#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  Mat *m = mat_ones(2, 3);

  printf("2x3 ones:\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
