#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"
#include <stdlib.h>

// Helper to compute P * A where P is a permutation
static void apply_row_perm(Mat *out, const Mat *A, const Perm *p) {
    size_t n = A->rows;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            out->data[i * n + j] = A->data[p->data[i] * n + j];
        }
    }
}

static void test_plu_2x2(void) {
    TEST_BEGIN("plu_2x2");

    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        4, 3,
        6, 3
    });

    Mat *L = mat_mat(2, 2);
    Mat *U = mat_mat(2, 2);
    Perm *p = mat_perm(2);

    mat_plu(A, L, U, p);

    // Verify P * A = L * U
    Mat *PA = mat_mat(2, 2);
    apply_row_perm(PA, A, p);

    Mat *LU = mat_rmul(L, U);

    for (size_t i = 0; i < 4; i++) {
        CHECK_FLOAT_EQ_TOL(PA->data[i], LU->data[i], 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_mat(PA);
    mat_free_mat(LU);
    mat_free_perm(p);

    TEST_END();
}

static void test_plu_3x3(void) {
    TEST_BEGIN("plu_3x3");

    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        0, 1, 4,
        5, 6, 0
    });

    Mat *L = mat_mat(3, 3);
    Mat *U = mat_mat(3, 3);
    Perm *p = mat_perm(3);

    mat_plu(A, L, U, p);

    // Verify P * A = L * U
    Mat *PA = mat_mat(3, 3);
    apply_row_perm(PA, A, p);

    Mat *LU = mat_rmul(L, U);

    for (size_t i = 0; i < 9; i++) {
        CHECK_FLOAT_EQ_TOL(PA->data[i], LU->data[i], 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_mat(PA);
    mat_free_mat(LU);
    mat_free_perm(p);

    TEST_END();
}

static void test_plu_random(size_t n, const char *name, mat_elem_t tol) {
    TEST_BEGIN(name);

    Mat *A = mat_mat(n, n);
    // Make diagonally dominant to ensure good conditioning
    for (size_t i = 0; i < n; i++) {
        mat_elem_t row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            mat_elem_t val = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
            A->data[i * n + j] = val;
            if (i != j) row_sum += MAT_FABS(val);
        }
        A->data[i * n + i] = row_sum + 1.0f;
    }

    Mat *L = mat_mat(n, n);
    Mat *U = mat_mat(n, n);
    Perm *p = mat_perm(n);

    mat_plu(A, L, U, p);

    // Verify P * A = L * U
    Mat *PA = mat_mat(n, n);
    apply_row_perm(PA, A, p);

    Mat *LU = mat_rmul(L, U);

    for (size_t i = 0; i < n * n; i++) {
        CHECK_FLOAT_EQ_TOL(PA->data[i], LU->data[i], tol);
    }

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_mat(PA);
    mat_free_mat(LU);
    mat_free_perm(p);

    TEST_END();
}

static void test_det_consistency(void) {
    TEST_BEGIN("det_consistency");

    // Verify mat_det (using mat_plu) gives same result as manual computation via mat_lu
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        6, 1, 1,
        4, -2, 5,
        2, 8, 7
    });

    // Get determinant via mat_det (uses mat_plu)
    mat_elem_t det_plu = mat_det(A);

    // Compute determinant manually via mat_lu (full pivoting)
    Mat *L = mat_mat(3, 3);
    Mat *U = mat_mat(3, 3);
    Perm *p = mat_perm(3);
    Perm *q = mat_perm(3);

    int swaps = mat_lu(A, L, U, p, q);
    mat_elem_t det_lu = (swaps % 2 == 0) ? 1 : -1;
    for (size_t i = 0; i < 3; i++) {
        det_lu *= U->data[i * 3 + i];
    }

    CHECK_FLOAT_EQ_TOL(det_plu, det_lu, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_perm(p);
    mat_free_perm(q);

    TEST_END();
}

int main(void) {
    srand(42);

    test_plu_2x2();
    test_plu_3x3();
    test_plu_random(10, "plu_random_10x10", 1e-4f);
    test_plu_random(50, "plu_random_50x50", 1e-4f);
    test_plu_random(100, "plu_random_100x100", 1e-3f);
    test_det_consistency();

    TEST_SUMMARY();
}
