#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Mat *m = mat_from(3, 3, vals);

  printf("matrix:\n");
  mat_print(m);

  Vec *d = mat_diag(m);
  printf("diagonal:\n");
  mat_print(d);

  mat_free_mat(m);
  mat_free_mat(d);
  return 0;
}
