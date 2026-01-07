#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Vec *v = mat_vec_from(4, vals);

  printf("vector:\n");
  mat_print(v);

  mat_free_mat(v);
  return 0;
}
