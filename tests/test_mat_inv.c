#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"
#include <stdlib.h>

static void test_inv_2x2(void) {
    TEST_BEGIN("inv_2x2");

    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        4, 7,
        2, 6
    });

    Mat *Ainv = mat_mat(2, 2);
    mat_inv(Ainv, A);

    // Check A * A^-1 = I
    Mat *I = mat_rmul(A, Ainv);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            CHECK_FLOAT_EQ_TOL(mat_at(I, i, j), expected, 1e-5f);
        }
    }

    mat_free_mat(A);
    mat_free_mat(Ainv);
    mat_free_mat(I);

    TEST_END();
}

static void test_inv_3x3(void) {
    TEST_BEGIN("inv_3x3");

    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        0, 1, 4,
        5, 6, 0
    });

    Mat *Ainv = mat_mat(3, 3);
    mat_inv(Ainv, A);

    // Check A * A^-1 = I
    Mat *I = mat_rmul(A, Ainv);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            CHECK_FLOAT_EQ_TOL(mat_at(I, i, j), expected, 1e-5f);
        }
    }

    mat_free_mat(A);
    mat_free_mat(Ainv);
    mat_free_mat(I);

    TEST_END();
}

static void test_inv_identity(void) {
    TEST_BEGIN("inv_identity_5x5");

    Mat *I = mat_reye(5);
    Mat *Iinv = mat_mat(5, 5);
    mat_inv(Iinv, I);

    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            CHECK_FLOAT_EQ_TOL(mat_at(Iinv, i, j), expected, 1e-5f);
        }
    }

    mat_free_mat(I);
    mat_free_mat(Iinv);

    TEST_END();
}

static void test_inv_random(size_t n, const char *name) {
    TEST_BEGIN(name);

    Mat *A = mat_mat(n, n);
    // Make diagonally dominant to ensure invertibility
    for (size_t i = 0; i < n; i++) {
        mat_elem_t row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            mat_elem_t val = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
            mat_set_at(A, i, j, val);
            if (i != j) row_sum += MAT_FABS(val);
        }
        mat_set_at(A, i, i, row_sum + 1.0f);
    }

    Mat *Ainv = mat_mat(n, n);
    mat_inv(Ainv, A);

    // Check A * A^-1 = I
    // Use looser tolerance for large matrices (float precision limits)
    Mat *I = mat_rmul(A, Ainv);
    mat_elem_t tol = (n >= 100) ? 1e-3f : 1e-4f;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            CHECK_FLOAT_EQ_TOL(mat_at(I, i, j), expected, tol);
        }
    }

    mat_free_mat(A);
    mat_free_mat(Ainv);
    mat_free_mat(I);

    TEST_END();
}

int main(void) {
    srand(42);

    test_inv_2x2();
    test_inv_3x3();
    test_inv_identity();
    test_inv_random(10, "inv_random_10x10");
    test_inv_random(50, "inv_random_50x50");
    test_inv_random(100, "inv_random_100x100");

    TEST_SUMMARY();
}
