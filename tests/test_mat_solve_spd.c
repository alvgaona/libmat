#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"
#include <stdlib.h>

// Helper: Create SPD matrix A = M * M^T + epsilon * I
static void make_spd(Mat *A, size_t n) {
  Mat *M = mat_mat(n, n);
  for (size_t i = 0; i < n * n; i++) {
    M->data[i] = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
  }
  // A = M * M^T
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += M->data[i * n + k] * M->data[j * n + k];
      }
      A->data[i * n + j] = sum;
    }
  }
  // Add epsilon * I for numerical stability
  for (size_t i = 0; i < n; i++) {
    A->data[i * n + i] += 0.1f;
  }
  mat_free_mat(M);
}

static void test_solve_spd_identity(void) {
  TEST_BEGIN("solve_spd_identity");

  // A = I (identity is SPD), so x = b
  Mat *A = mat_reye(3);
  Vec *b = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});
  Vec *x = mat_vec(3);

  int ret = mat_solve_spd(x, A, b);
  CHECK(ret == 0);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-6f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-6f);
  CHECK_FLOAT_EQ_TOL(x->data[2], 3.0f, 1e-6f);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_spd_2x2(void) {
  TEST_BEGIN("solve_spd_2x2");

  // A = [4, 2; 2, 3] is SPD (eigenvalues ~1.27 and ~5.73)
  Mat *A = mat_from(2, 2, (mat_elem_t[]){
    4, 2,
    2, 3
  });
  // x = [1, 2], b = A*x = [4+4, 2+6] = [8, 8]
  Vec *b = mat_vec_from(2, (mat_elem_t[]){8, 8});
  Vec *x = mat_vec(2);

  int ret = mat_solve_spd(x, A, b);
  CHECK(ret == 0);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-5f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-5f);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_spd_3x3(void) {
  TEST_BEGIN("solve_spd_3x3");

  // Construct SPD matrix: A = M * M^T where M = [2,1,0; 1,2,1; 0,1,2]
  // A = [5,4,1; 4,6,4; 1,4,5]
  Mat *A = mat_from(3, 3, (mat_elem_t[]){
    5, 4, 1,
    4, 6, 4,
    1, 4, 5
  });
  // x = [1, 2, 3], b = A*x = [5+8+3, 4+12+12, 1+8+15] = [16, 28, 24]
  Vec *b = mat_vec_from(3, (mat_elem_t[]){16, 28, 24});
  Vec *x = mat_vec(3);

  int ret = mat_solve_spd(x, A, b);
  CHECK(ret == 0);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-4f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-4f);
  CHECK_FLOAT_EQ_TOL(x->data[2], 3.0f, 1e-4f);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_spd_not_positive_definite(void) {
  TEST_BEGIN("solve_spd_not_positive_definite");

  // A = [1, 2; 2, 1] has eigenvalues -1 and 3, not positive definite
  Mat *A = mat_from(2, 2, (mat_elem_t[]){
    1, 2,
    2, 1
  });
  Vec *b = mat_vec_from(2, (mat_elem_t[]){1, 1});
  Vec *x = mat_vec(2);

  int ret = mat_solve_spd(x, A, b);
  CHECK(ret == -1);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_spd_verify_residual(size_t n, const char *name, mat_elem_t tol) {
  TEST_BEGIN(name);

  // Generate random SPD matrix
  Mat *A = mat_mat(n, n);
  make_spd(A, n);

  // Generate random b
  Vec *b = mat_vec(n);
  for (size_t i = 0; i < n; i++) {
    b->data[i] = (mat_elem_t)(rand() % 100) / 10.0f;
  }

  Vec *x = mat_vec(n);
  int ret = mat_solve_spd(x, A, b);
  CHECK(ret == 0);

  // Verify: A * x should equal b
  Vec *Ax = mat_vec(n);
  mat_gemv(Ax, 1.0f, A, x, 0.0f);

  for (size_t i = 0; i < n; i++) {
    CHECK_FLOAT_EQ_TOL(Ax->data[i], b->data[i], tol);
  }

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);
  mat_free_mat(Ax);

  TEST_END();
}

static void test_solve_spd_1x1(void) {
  TEST_BEGIN("solve_spd_1x1");

  Mat *A = mat_from(1, 1, (mat_elem_t[]){4.0f});
  Vec *b = mat_vec_from(1, (mat_elem_t[]){12.0f});
  Vec *x = mat_vec(1);

  int ret = mat_solve_spd(x, A, b);
  CHECK(ret == 0);

  CHECK_FLOAT_EQ_TOL(x->data[0], 3.0f, 1e-6f);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

int main(void) {
  srand(42);

  test_solve_spd_1x1();
  test_solve_spd_identity();
  test_solve_spd_2x2();
  test_solve_spd_3x3();
  test_solve_spd_not_positive_definite();
  test_solve_spd_verify_residual(10, "solve_spd_random_10x10", 1e-4f);
  test_solve_spd_verify_residual(50, "solve_spd_random_50x50", 1e-3f);
  test_solve_spd_verify_residual(100, "solve_spd_random_100x100", 1e-2f);

  TEST_SUMMARY();
}
