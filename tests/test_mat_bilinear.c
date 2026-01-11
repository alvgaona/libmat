#include <stdio.h>
#include <math.h>
#define MAT_IMPLEMENTATION
#include "mat.h"

int main(void) {
  int passed = 0, failed = 0;

  // Test 1: Simple 2x2 case
  // x = [1, 2], A = [[1, 2], [3, 4]], y = [5, 6]
  // x^T * A * y = [1, 2] * [[1, 2], [3, 4]] * [5, 6]^T
  //             = [1, 2] * [1*5 + 2*6, 3*5 + 4*6]^T
  //             = [1, 2] * [17, 39]^T
  //             = 1*17 + 2*39 = 17 + 78 = 95
  {
    Vec *x = mat_vnew({1.0f, 2.0f});
    Mat *A = mat_new(2, {{1.0f, 2.0f}, {3.0f, 4.0f}});
    Vec *y = mat_vnew({5.0f, 6.0f});

    mat_elem_t result = mat_bilinear(x, A, y);

    if (fabsf(result - 95.0f) < 1e-5f) {
      printf("PASS: mat_bilinear 2x2\n");
      passed++;
    } else {
      printf("FAIL: mat_bilinear 2x2 (expected 95, got %f)\n", result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(A);
    mat_free_mat(y);
  }

  // Test 2: Quadratic form x^T * A * x with identity using mat_bilinear
  // x^T * I * x = x^T * x = ||x||^2
  {
    Vec *x = mat_vnew({1.0f, 2.0f, 3.0f});
    Mat *I = mat_reye(3);

    mat_elem_t result = mat_bilinear(x, I, x);
    mat_elem_t expected = 1.0f + 4.0f + 9.0f; // 14

    if (fabsf(result - expected) < 1e-5f) {
      printf("PASS: mat_bilinear quadratic form with identity\n");
      passed++;
    } else {
      printf("FAIL: mat_bilinear quadratic form (expected %f, got %f)\n", expected, result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(I);
  }

  // Test 2b: mat_quadform convenience function
  {
    Vec *x = mat_vnew({1.0f, 2.0f, 3.0f});
    Mat *I = mat_reye(3);

    mat_elem_t result = mat_quadform(x, I);
    mat_elem_t expected = 1.0f + 4.0f + 9.0f; // 14

    if (fabsf(result - expected) < 1e-5f) {
      printf("PASS: mat_quadform with identity\n");
      passed++;
    } else {
      printf("FAIL: mat_quadform (expected %f, got %f)\n", expected, result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(I);
  }

  // Test 2c: mat_quadform with non-trivial matrix
  // x = [1, 2], A = [[2, 1], [1, 3]]
  // x^T * A * x = [1, 2] * [[2, 1], [1, 3]] * [1, 2]^T
  //             = [1, 2] * [2+2, 1+6]^T = [1, 2] * [4, 7]^T = 4 + 14 = 18
  {
    Vec *x = mat_vnew({1.0f, 2.0f});
    Mat *A = mat_new(2, {{2.0f, 1.0f}, {1.0f, 3.0f}});

    mat_elem_t result = mat_quadform(x, A);

    if (fabsf(result - 18.0f) < 1e-5f) {
      printf("PASS: mat_quadform with symmetric matrix\n");
      passed++;
    } else {
      printf("FAIL: mat_quadform symmetric (expected 18, got %f)\n", result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(A);
  }

  // Test 3: Non-square matrix (3x4)
  // x = [1, 2, 3], A = 3x4 matrix, y = [1, 2, 3, 4]
  {
    Vec *x = mat_vnew({1.0f, 2.0f, 3.0f});
    Mat *A = mat_new(4, {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {5.0f, 6.0f, 7.0f, 8.0f},
      {9.0f, 10.0f, 11.0f, 12.0f}
    });
    Vec *y = mat_vnew({1.0f, 2.0f, 3.0f, 4.0f});

    // Row 0 dot y = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    // Row 1 dot y = 5*1 + 6*2 + 7*3 + 8*4 = 5 + 12 + 21 + 32 = 70
    // Row 2 dot y = 9*1 + 10*2 + 11*3 + 12*4 = 9 + 20 + 33 + 48 = 110
    // result = 1*30 + 2*70 + 3*110 = 30 + 140 + 330 = 500

    mat_elem_t result = mat_bilinear(x, A, y);

    if (fabsf(result - 500.0f) < 1e-5f) {
      printf("PASS: mat_bilinear 3x4 non-square\n");
      passed++;
    } else {
      printf("FAIL: mat_bilinear 3x4 (expected 500, got %f)\n", result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(A);
    mat_free_mat(y);
  }

  // Test 4: Larger matrix to exercise SIMD
  {
    size_t n = 256;
    Vec *x = mat_vec(n);
    Mat *A = mat_mat(n, n);
    Vec *y = mat_vec(n);

    // Fill with simple values
    for (size_t i = 0; i < n; i++) {
      x->data[i] = 1.0f;
      y->data[i] = 1.0f;
      for (size_t j = 0; j < n; j++) {
        A->data[i * n + j] = 1.0f;
      }
    }

    // x^T * A * y where all are 1s
    // Each row dot y = n (sum of n ones)
    // result = sum of n rows, each contributing n = n * n = n^2
    mat_elem_t result = mat_bilinear(x, A, y);
    mat_elem_t expected = (mat_elem_t)(n * n);

    if (fabsf(result - expected) < 1e-2f) {
      printf("PASS: mat_bilinear 256x256 SIMD\n");
      passed++;
    } else {
      printf("FAIL: mat_bilinear 256x256 (expected %f, got %f)\n", expected, result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(A);
    mat_free_mat(y);
  }

  // Test 5: Compare with manual gemv + dot
  {
    size_t m = 64, n = 128;
    Vec *x = mat_vec(m);
    Mat *A = mat_mat(m, n);
    Vec *y = mat_vec(n);

    for (size_t i = 0; i < m; i++) x->data[i] = (mat_elem_t)(i + 1);
    for (size_t i = 0; i < n; i++) y->data[i] = (mat_elem_t)(i + 1);
    for (size_t i = 0; i < m * n; i++) A->data[i] = (mat_elem_t)((i % 10) + 1);

    // Manual: temp = A * y, result = x^T * temp
    Vec *temp = mat_vec(m);
    mat_gemv(temp, 1.0f, A, y, 0.0f);
    mat_elem_t expected = mat_dot(x, temp);

    mat_elem_t result = mat_bilinear(x, A, y);

    if (fabsf(result - expected) / fabsf(expected) < 1e-4f) {
      printf("PASS: mat_bilinear matches gemv+dot\n");
      passed++;
    } else {
      printf("FAIL: mat_bilinear vs gemv+dot (expected %f, got %f)\n", expected, result);
      failed++;
    }

    mat_free_mat(x);
    mat_free_mat(A);
    mat_free_mat(y);
    mat_free_mat(temp);
  }

  printf("\n%d passed, %d failed\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
