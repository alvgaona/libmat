#include <stdio.h>

// Define MATDEF as static inline for potential inlining
#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("With MATDEF=static inline:\n");
  mat_print(m);

  mat_scale(m, 2);
  printf("scaled by 2:\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
