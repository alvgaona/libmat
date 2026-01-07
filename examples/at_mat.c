#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("matrix:\n");
  mat_print(m);

  printf("m[0,0] = %g\n", mat_at(m, 0, 0));
  printf("m[0,1] = %g\n", mat_at(m, 0, 1));
  printf("m[1,0] = %g\n", mat_at(m, 1, 0));
  printf("m[1,1] = %g\n", mat_at(m, 1, 1));

  mat_free_mat(m);
  return 0;
}
