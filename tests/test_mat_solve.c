#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"
#include <stdlib.h>

static void test_solve_identity(void) {
    TEST_BEGIN("solve_identity");

    // A = I, so x = b
    Mat *A = mat_reye(3);
    Vec *b = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});
    Vec *x = mat_vec(3);

    mat_solve(x, A, b);

    CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-6f);
    CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-6f);
    CHECK_FLOAT_EQ_TOL(x->data[2], 3.0f, 1e-6f);

    mat_free_mat(A);
    mat_free_mat(b);
    mat_free_mat(x);

    TEST_END();
}

static void test_solve_2x2(void) {
    TEST_BEGIN("solve_2x2");

    // 2x + 3y = 8
    // 4x + 5y = 14
    // Solution: x = 1, y = 2
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        2, 3,
        4, 5
    });
    Vec *b = mat_vec_from(2, (mat_elem_t[]){8, 14});
    Vec *x = mat_vec(2);

    mat_solve(x, A, b);

    CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(b);
    mat_free_mat(x);

    TEST_END();
}

static void test_solve_3x3(void) {
    TEST_BEGIN("solve_3x3");

    // System with known solution
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        0, 1, 4,
        5, 6, 0
    });
    // x = [1, 2, 3], so b = A*x = [1+4+9, 0+2+12, 5+12+0] = [14, 14, 17]
    Vec *b = mat_vec_from(3, (mat_elem_t[]){14, 14, 17});
    Vec *x = mat_vec(3);

    mat_solve(x, A, b);

    CHECK_FLOAT_EQ_TOL(x->data[0], 1.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(x->data[1], 2.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(x->data[2], 3.0f, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(b);
    mat_free_mat(x);

    TEST_END();
}

static void test_solve_verify_residual(size_t n, const char *name, mat_elem_t tol) {
    TEST_BEGIN(name);

    // Generate random diagonally dominant matrix (well-conditioned)
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        mat_elem_t row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            mat_elem_t val = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
            A->data[i * n + j] = val;
            if (i != j) row_sum += MAT_FABS(val);
        }
        A->data[i * n + i] = row_sum + 1.0f;
    }

    // Generate random b
    Vec *b = mat_vec(n);
    for (size_t i = 0; i < n; i++) {
        b->data[i] = (mat_elem_t)(rand() % 100) / 10.0f;
    }

    Vec *x = mat_vec(n);
    mat_solve(x, A, b);

    // Verify: A * x should equal b
    // Compute residual: r = A * x - b
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

static void test_solve_1x1(void) {
    TEST_BEGIN("solve_1x1");

    Mat *A = mat_from(1, 1, (mat_elem_t[]){5.0f});
    Vec *b = mat_vec_from(1, (mat_elem_t[]){15.0f});
    Vec *x = mat_vec(1);

    mat_solve(x, A, b);

    CHECK_FLOAT_EQ_TOL(x->data[0], 3.0f, 1e-6f);

    mat_free_mat(A);
    mat_free_mat(b);
    mat_free_mat(x);

    TEST_END();
}

int main(void) {
    srand(42);

    test_solve_1x1();
    test_solve_identity();
    test_solve_2x2();
    test_solve_3x3();
    test_solve_verify_residual(10, "solve_random_10x10", 1e-4f);
    test_solve_verify_residual(50, "solve_random_50x50", 1e-3f);
    test_solve_verify_residual(100, "solve_random_100x100", 1e-2f);

    TEST_SUMMARY();
}
