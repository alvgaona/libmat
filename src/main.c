#include <stdio.h>
#include <stdlib.h>

#include "mat.h"
#include "util.h"

int main() {
  size_t values[2][2] = {{1, 0}, {0, 1}};

  Mat *m1 = create_mat(2, 2);
  init_mat(m1, (mat_elem_t *)values);

  Mat *m2 = create_mat(2, 2);
  init_mat(m2, (mat_elem_t *)values);

  Mat *result = sub_mat_mat(m1, m2);

  print_mat(result);

  Mat *m3 = ones(3, 3);
  Mat *m4 = zeros(3, 3);

  print_mat(m3);
  print_mat(m4);

  Mat *m5 = identity(4);

  print_mat(m5);
  
  free(m1);
  free(m2);
  free(m3);
  free(m4);
  free(m5);
  free(result);
  
  return 0;
}

