#include <stdio.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);

  printf("original:\n");
  mat_print(m);

  Mat *deep = mat_deep_copy(m);
  printf("deep copy:\n");
  mat_print(deep);

  // Modify original to show independence
  m->data[0] = 99;
  printf("after modifying original[0,0] = 99:\n");
  printf("original:\n");
  mat_print(m);
  printf("deep copy (unchanged):\n");
  mat_print(deep);

  mat_free_mat(m);
  mat_free_mat(deep);
  return 0;
}
