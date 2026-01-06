#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  mat_elem_t vals1[] = {1, 2, 3, 4};
  mat_elem_t vals2[] = {5, 6, 7, 8};
  Mat *m1 = mat_from(2, 2, vals1);
  Mat *m2 = mat_from(2, 2, vals2);
  Mat *out = mat_mat(2, 2);

  mat_add(out, m1, m2);

  printf("m1 + m2 into out:\n");
  mat_print(out);

  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(out);
  return 0;
}
