#define MAT_IMPLEMENTATION
#include "mat.h"
#include <stdio.h>

// Helper to compute P * A where P is a permutation
static void apply_row_perm(Mat *out, const Mat *A, const Perm *p) {
    size_t n = A->rows;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            out->data[i * n + j] = A->data[p->data[i] * n + j];
        }
    }
}

static int test_plu_2x2(void) {
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

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 4; i++) {
        mat_elem_t err = fabsf(PA->data[i] - LU->data[i]);
        if (err > max_err) max_err = err;
    }

    printf("2x2: P * A = L * U, max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_mat(PA);
    mat_free_mat(LU);
    mat_free_perm(p);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_plu_3x3(void) {
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

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 9; i++) {
        mat_elem_t err = fabsf(PA->data[i] - LU->data[i]);
        if (err > max_err) max_err = err;
    }

    printf("3x3: P * A = L * U, max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_mat(PA);
    mat_free_mat(LU);
    mat_free_perm(p);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_plu_random(size_t n) {
    Mat *A = mat_mat(n, n);
    // Make diagonally dominant to ensure good conditioning
    for (size_t i = 0; i < n; i++) {
        mat_elem_t row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            mat_elem_t val = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
            A->data[i * n + j] = val;
            if (i != j) row_sum += fabsf(val);
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

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < n * n; i++) {
        mat_elem_t err = fabsf(PA->data[i] - LU->data[i]);
        if (err > max_err) max_err = err;
    }

    // Tolerance scales with matrix size for single precision
    mat_elem_t tol = (n <= 50) ? 1e-4f : 1e-3f;
    printf("%zux%zu random: P * A = L * U, max error = %e %s\n", n, n, max_err, max_err < tol ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_mat(PA);
    mat_free_mat(LU);
    mat_free_perm(p);

    return max_err < tol ? 0 : 1;
}

static int test_det_consistency(void) {
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

    mat_elem_t err = fabsf(det_plu - det_lu);
    printf("det consistency: plu=%.4f, lu=%.4f, error = %e %s\n",
           det_plu, det_lu, err, err < 1e-4f ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(U);
    mat_free_perm(p);
    mat_free_perm(q);

    return err < 1e-4f ? 0 : 1;
}

int main(void) {
    srand(42);
    int failures = 0;

    failures += test_plu_2x2();
    failures += test_plu_3x3();
    failures += test_plu_random(10);
    failures += test_plu_random(50);
    failures += test_plu_random(100);
    failures += test_det_consistency();

    printf("\n%s\n", failures == 0 ? "All tests passed!" : "Some tests failed!");
    return failures;
}
