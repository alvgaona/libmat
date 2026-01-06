#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  // Create two 2x2 matrices
  Mat *m1 = mat_mat(2, 2);
  Mat *m2 = mat_mat(2, 2);

  // Initialize with values
  mat_elem_t vals1[] = {1, 2, 3, 4};
  mat_elem_t vals2[] = {5, 6, 7, 8};
  mat_init(m1, vals1);
  mat_init(m2, vals2);

  // Add matrices
  Mat *result = mat_radd(m1, m2);

  // Print results
  printf("m1:\n");
  mat_print(m1);
  printf("m2:\n");
  mat_print(m2);
  printf("m1 + m2:\n");
  mat_print(result);

  // Clean up
  mat_free_mat(m1);
  mat_free_mat(m2);
  mat_free_mat(result);

  return 0;
}
