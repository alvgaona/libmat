#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  // Create a 2x2 matrix
  Mat *m = mat_mat(2, 2);

  // Initialize with values
  mat_elem_t vals[] = {1, 2, 3, 4};
  mat_init(m, vals);

  // Transpose matrix
  Mat *result = mat_rt(m);

  // Print results
  printf("m:\n");
  mat_print(m);
  printf("m^T:\n");
  mat_print(result);

  // Clean up
  mat_free_mat(m);
  mat_free_mat(result);

  return 0;
}
