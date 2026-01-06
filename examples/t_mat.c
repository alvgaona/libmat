#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  mat_elem_t vals[] = {1, 2, 3, 4};
  Mat *m = mat_from(2, 2, vals);
  Mat *out = mat_mat(2, 2);

  printf("m:\n");
  mat_print(m);

  mat_t(out, m);
  printf("transposed into out:\n");
  mat_print(out);

  mat_free_mat(m);
  mat_free_mat(out);
  return 0;
}
