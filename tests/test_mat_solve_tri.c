#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"
#include <stdlib.h>

static void test_solve_tril_2x2(void) {
  TEST_BEGIN("solve_tril_2x2");

  // L = [2, 0; 3, 4], x = [1, 2], b = L*x = [2, 11]
  Mat *L = mat_from(2, 2, (mat_elem_t[]){
    2, 0,
    3, 4
  });
  Vec *b = mat_vec_from(2, (mat_elem_t[]){2, 11});
  Vec *x = mat_vec(2);

  mat_solve_tril(x, L, b);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-6f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-6f);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_tril_unit_2x2(void) {
  TEST_BEGIN("solve_tril_unit_2x2");

  // L = [1, 0; 3, 1] (unit diagonal), x = [1, 2], b = L*x = [1, 5]
  Mat *L = mat_from(2, 2, (mat_elem_t[]){
    1, 0,
    3, 1
  });
  Vec *b = mat_vec_from(2, (mat_elem_t[]){1, 5});
  Vec *x = mat_vec(2);

  mat_solve_tril_unit(x, L, b);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-6f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-6f);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_triu_2x2(void) {
  TEST_BEGIN("solve_triu_2x2");

  // U = [2, 3; 0, 4], x = [1, 2], b = U*x = [8, 8]
  Mat *U = mat_from(2, 2, (mat_elem_t[]){
    2, 3,
    0, 4
  });
  Vec *b = mat_vec_from(2, (mat_elem_t[]){8, 8});
  Vec *x = mat_vec(2);

  mat_solve_triu(x, U, b);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-6f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-6f);

  mat_free_mat(U);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_trilt_2x2(void) {
  TEST_BEGIN("solve_trilt_2x2");

  // L = [2, 0; 3, 4], L^T = [2, 3; 0, 4]
  // x = [1, 2], b = L^T * x = [8, 8]
  Mat *L = mat_from(2, 2, (mat_elem_t[]){
    2, 0,
    3, 4
  });
  Vec *b = mat_vec_from(2, (mat_elem_t[]){8, 8});
  Vec *x = mat_vec(2);

  mat_solve_trilt(x, L, b);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-6f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-6f);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_tril_3x3(void) {
  TEST_BEGIN("solve_tril_3x3");

  Mat *L = mat_from(3, 3, (mat_elem_t[]){
    2, 0, 0,
    1, 3, 0,
    4, 2, 5
  });
  // x = [1, 2, 3], b = L*x = [2, 7, 23]
  Vec *b = mat_vec_from(3, (mat_elem_t[]){2, 7, 23});
  Vec *x = mat_vec(3);

  mat_solve_tril(x, L, b);

  CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-5f);
  CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-5f);
  CHECK_FLOAT_EQ_TOL(x->data[2], 3.0f, 1e-5f);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);

  TEST_END();
}

static void test_solve_trilt_random(size_t n, const char *name, mat_elem_t tol) {
  TEST_BEGIN(name);

  // Create well-conditioned lower triangular (diagonally dominant)
  Mat *L = mat_mat(n, n);
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < i; j++) {
      mat_elem_t val = (mat_elem_t)(rand() % 100) / 100.0f - 0.5f;
      L->data[i * n + j] = val;
      row_sum += (val > 0 ? val : -val);
    }
    L->data[i * n + i] = row_sum + 1.0f; // Diagonal dominant
  }

  // Random b
  Vec *b = mat_vec(n);
  for (size_t i = 0; i < n; i++) {
    b->data[i] = (mat_elem_t)(rand() % 100) / 10.0f;
  }

  Vec *x = mat_vec(n);
  mat_solve_trilt(x, L, b);

  // Verify: L^T * x should equal b
  // L^T[i,j] = L[j,i]
  Vec *Lt_x = mat_vec(n);
  for (size_t i = 0; i < n; i++) {
    mat_elem_t sum = 0;
    for (size_t j = i; j < n; j++) {
      sum += L->data[j * n + i] * x->data[j];
    }
    Lt_x->data[i] = sum;
  }

  for (size_t i = 0; i < n; i++) {
    CHECK_FLOAT_EQ_TOL(Lt_x->data[i], b->data[i], tol);
  }

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);
  mat_free_mat(Lt_x);

  TEST_END();
}

int main(void) {
  srand(42);

  test_solve_tril_2x2();
  test_solve_tril_unit_2x2();
  test_solve_triu_2x2();
  test_solve_trilt_2x2();
  test_solve_tril_3x3();
  test_solve_trilt_random(10, "solve_trilt_random_10x10", 1e-4f);
  test_solve_trilt_random(50, "solve_trilt_random_50x50", 1e-3f);
  test_solve_trilt_random(100, "solve_trilt_random_100x100", 1e-2f);

  TEST_SUMMARY();
}
