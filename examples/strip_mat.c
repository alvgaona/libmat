#include <stdio.h>

// Enable shorter function names (mat instead of mat_mat, etc.)
#define MAT_STRIP_PREFIX
#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  // Using stripped names: mat() instead of mat_mat()
  Mat *m = mat(2, 2);

  // zeros() instead of mat_zeros()
  Mat *z = zeros(2, 2);

  // ones() instead of mat_ones()
  Mat *o = ones(2, 2);

  // eye() instead of mat_reye()
  Mat *i = eye(3);

  printf("mat(2, 2):\n");
  mat_print(m);

  printf("zeros(2, 2):\n");
  mat_print(z);

  printf("ones(2, 2):\n");
  mat_print(o);

  printf("eye(3):\n");
  mat_print(i);

  // rt() instead of mat_rt()
  Mat *tr = rt(o);
  printf("rt(ones):\n");
  mat_print(tr);

  // free_mat() instead of mat_free_mat()
  free_mat(m);
  free_mat(z);
  free_mat(o);
  free_mat(i);
  free_mat(tr);

  return 0;
}
