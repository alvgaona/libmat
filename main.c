#include <stdio.h>
#include <stdlib.h>

#include "mat.h"
#include "util.h"

int main() {
  float values[2][2] = {{1, -1}, {1, -1}};

  Mat *m1 = mat_mat(2, 2);
  mat_init(m1, (mat_elem_t *)values);

  Mat *m2 = mat_mat(2, 2);
  mat_init(m2, (mat_elem_t *)values);

  Mat *result = mat_rsub(m1, m2);

  mat_print(result);

  Mat *m3 = mat_ones(3, 3);
  Mat *m4 = mat_zeros(3, 3);

  mat_print(m3);
  mat_print(m4);

  Mat *m5 = mat_eye(4);

  mat_print(m5);

  Mat *mt1 = mat_mat(2, 2);
  mat_t(mt1, m1);

  mat_print(mt1);
  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(m3);
  mat_free_mat(m4);
  mat_free_mat(m5);
  mat_free_mat(result);
  
  return 0;
}

