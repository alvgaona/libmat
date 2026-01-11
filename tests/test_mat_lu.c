#define MAT_IMPLEMENTATION
#include "mat.h"
#include <stdio.h>

static int test_lu_3x3(void) {
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        2, -1, -2,
        -4, 6, 3,
        -4, -2, 8
    });

    Mat *L = mat_mat(3, 3);
    Mat *U = mat_mat(3, 3);
    Perm *p = mat_perm(3);
    Perm *q = mat_perm(3);

    int swaps = mat_lu(A, L, U, p, q);
    printf("3x3: %d swaps\n", swaps);

    // Verify P * A * Q = L * U
    Mat *PAQ = mat_mat(3, 3);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            PAQ->data[i * 3 + j] = A->data[p->data[i] * 3 + q->data[j]];
        }
    }

    Mat *LU = mat_rmul(L, U);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 9; i++) {
        mat_elem_t err = fabsf(LU->data[i] - PAQ->data[i]);
        if (err > max_err) max_err = err;
    }

    printf("  P*A*Q vs L*U max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    printf("  L:\n");
    for (size_t i = 0; i < 3; i++) {
        printf("    ");
        for (size_t j = 0; j < 3; j++) printf("%8.4f ", L->data[i * 3 + j]);
        printf("\n");
    }
    printf("  U:\n");
    for (size_t i = 0; i < 3; i++) {
        printf("    ");
        for (size_t j = 0; j < 3; j++) printf("%8.4f ", U->data[i * 3 + j]);
        printf("\n");
    }

    mat_free_mat(A); mat_free_mat(L); mat_free_mat(U);
    mat_free_mat(PAQ); mat_free_mat(LU);
    mat_free_perm(p); mat_free_perm(q);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_lu_4x4(void) {
    Mat *A = mat_from(4, 4, (mat_elem_t[]){
        4, 3, 2, 1,
        3, 4, 3, 2,
        2, 3, 4, 3,
        1, 2, 3, 4
    });

    Mat *L = mat_mat(4, 4);
    Mat *U = mat_mat(4, 4);
    Perm *p = mat_perm(4);
    Perm *q = mat_perm(4);

    int swaps = mat_lu(A, L, U, p, q);
    printf("4x4: %d swaps\n", swaps);

    Mat *PAQ = mat_mat(4, 4);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            PAQ->data[i * 4 + j] = A->data[p->data[i] * 4 + q->data[j]];
        }
    }

    Mat *LU = mat_rmul(L, U);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 16; i++) {
        mat_elem_t err = fabsf(LU->data[i] - PAQ->data[i]);
        if (err > max_err) max_err = err;
    }

    printf("  P*A*Q vs L*U max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A); mat_free_mat(L); mat_free_mat(U);
    mat_free_mat(PAQ); mat_free_mat(LU);
    mat_free_perm(p); mat_free_perm(q);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_lu_random(size_t n) {
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n * n; i++) {
        A->data[i] = (mat_elem_t)(rand() % 200 - 100) / 10.0f;
    }

    Mat *L = mat_mat(n, n);
    Mat *U = mat_mat(n, n);
    Perm *p = mat_perm(n);
    Perm *q = mat_perm(n);

    int swaps = mat_lu(A, L, U, p, q);

    Mat *PAQ = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            PAQ->data[i * n + j] = A->data[p->data[i] * n + q->data[j]];
        }
    }

    Mat *LU = mat_rmul(L, U);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < n * n; i++) {
        mat_elem_t err = fabsf(LU->data[i] - PAQ->data[i]);
        if (err > max_err) max_err = err;
    }

    printf("%zux%zu: %d swaps, max error = %e %s\n", n, n, swaps, max_err, max_err < 1e-4f ? "OK" : "FAIL");

    mat_free_mat(A); mat_free_mat(L); mat_free_mat(U);
    mat_free_mat(PAQ); mat_free_mat(LU);
    mat_free_perm(p); mat_free_perm(q);

    return max_err < 1e-4f ? 0 : 1;
}

int main(void) {
    int failures = 0;

    failures += test_lu_3x3();
    failures += test_lu_4x4();
    failures += test_lu_random(10);
    failures += test_lu_random(50);
    failures += test_lu_random(100);

    printf("\n%s\n", failures == 0 ? "All tests passed!" : "Some tests failed!");
    return failures;
}
