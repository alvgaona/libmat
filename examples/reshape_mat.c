#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  // Create a 2x2 matrix
  Mat *m = mat_mat(2, 2);

  // Initialize with values
  mat_elem_t vals[] = {1, 2, 3, 4};
  mat_init(m, vals);

  // Print original
  printf("m (2x2):\n");
  mat_print(m);
  printf("\n");
  // Reshape to 4x1
  mat_reshape(m, 4, 1);

  // Print reshaped
  printf("m (4x1):\n");
  mat_print(m);

  // Clean up
  mat_free_mat(m);

  return 0;
}
