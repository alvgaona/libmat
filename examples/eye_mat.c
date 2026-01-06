#include <stdio.h>

#include "mat_util.h"
#include "mat.h"

int main() {
  Mat *m = mat_eye(3);

  printf("3x3 identity:\n");
  mat_print(m);

  mat_free_mat(m);
  return 0;
}
