#include <stdio.h>
#define MAT_IMPLEMENTATION
#include "mat.h"

int main(void) {
  int passed = 0, failed = 0;

  // Test mat_cross
  {
    Vec *v1 = mat_vnew({1.0f, 0.0f, 0.0f});
    Vec *v2 = mat_vnew({0.0f, 1.0f, 0.0f});
    Vec *out = mat_vec(3);

    mat_cross(out, v1, v2);

    // i x j = k
    Vec *expected = mat_vnew({0.0f, 0.0f, 1.0f});
    if (mat_equals(out, expected)) {
      printf("PASS: mat_cross i x j = k\n");
      passed++;
    } else {
      printf("FAIL: mat_cross i x j = k\n");
      mat_print(out);
      failed++;
    }

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);
  }

  // Test mat_cross - j x i = -k
  {
    Vec *v1 = mat_vnew({0.0f, 1.0f, 0.0f});
    Vec *v2 = mat_vnew({1.0f, 0.0f, 0.0f});
    Vec *out = mat_vec(3);

    mat_cross(out, v1, v2);

    Vec *expected = mat_vnew({0.0f, 0.0f, -1.0f});
    if (mat_equals(out, expected)) {
      printf("PASS: mat_cross j x i = -k\n");
      passed++;
    } else {
      printf("FAIL: mat_cross j x i = -k\n");
      mat_print(out);
      failed++;
    }

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);
  }

  // Test mat_cross - general case
  {
    Vec *v1 = mat_vnew({1.0f, 2.0f, 3.0f});
    Vec *v2 = mat_vnew({4.0f, 5.0f, 6.0f});
    Vec *out = mat_vec(3);

    mat_cross(out, v1, v2);

    // (1,2,3) x (4,5,6) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
    Vec *expected = mat_vnew({-3.0f, 6.0f, -3.0f});
    if (mat_equals(out, expected)) {
      printf("PASS: mat_cross general case\n");
      passed++;
    } else {
      printf("FAIL: mat_cross general case\n");
      mat_print(out);
      failed++;
    }

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);
  }

  // Test mat_outer - 2x3
  {
    Vec *v1 = mat_vnew({1.0f, 2.0f});
    Vec *v2 = mat_vnew({3.0f, 4.0f, 5.0f});
    Mat *out = mat_mat(2, 3);

    mat_outer(out, v1, v2);

    // [1]   [3 4 5]   [3  4  5]
    // [2] *         = [6  8  10]
    Mat *expected = mat_new(3, {
      {3.0f, 4.0f, 5.0f},
      {6.0f, 8.0f, 10.0f}
    });

    if (mat_equals(out, expected)) {
      printf("PASS: mat_outer 2x3\n");
      passed++;
    } else {
      printf("FAIL: mat_outer 2x3\n");
      mat_print(out);
      failed++;
    }

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);
  }

  // Test mat_outer - 3x2
  {
    Vec *v1 = mat_vnew({1.0f, 2.0f, 3.0f});
    Vec *v2 = mat_vnew({4.0f, 5.0f});
    Mat *out = mat_mat(3, 2);

    mat_outer(out, v1, v2);

    Mat *expected = mat_new(2, {
      {4.0f, 5.0f},
      {8.0f, 10.0f},
      {12.0f, 15.0f}
    });

    if (mat_equals(out, expected)) {
      printf("PASS: mat_outer 3x2\n");
      passed++;
    } else {
      printf("FAIL: mat_outer 3x2\n");
      mat_print(out);
      failed++;
    }

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);
  }

  printf("\n%d passed, %d failed\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
