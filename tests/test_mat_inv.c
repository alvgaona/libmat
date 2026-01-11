#define MAT_IMPLEMENTATION
#include "mat.h"
#include <stdio.h>

static int test_inv_2x2(void) {
    // A = [[4, 7], [2, 6]]
    // A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        4, 7,
        2, 6
    });

    Mat *Ainv = mat_mat(2, 2);
    mat_inv(Ainv, A);

    // Check A * A^-1 = I
    Mat *I = mat_rmul(A, Ainv);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            mat_elem_t err = fabsf(I->data[i * 2 + j] - expected);
            if (err > max_err) max_err = err;
        }
    }

    printf("2x2: A * A^-1 = I, max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(Ainv);
    mat_free_mat(I);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_inv_3x3(void) {
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        0, 1, 4,
        5, 6, 0
    });

    Mat *Ainv = mat_mat(3, 3);
    mat_inv(Ainv, A);

    // Check A * A^-1 = I
    Mat *I = mat_rmul(A, Ainv);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            mat_elem_t err = fabsf(I->data[i * 3 + j] - expected);
            if (err > max_err) max_err = err;
        }
    }

    printf("3x3: A * A^-1 = I, max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(Ainv);
    mat_free_mat(I);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_inv_identity(void) {
    // I^-1 = I
    Mat *I = mat_reye(5);
    Mat *Iinv = mat_mat(5, 5);
    mat_inv(Iinv, I);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            mat_elem_t err = fabsf(Iinv->data[i * 5 + j] - expected);
            if (err > max_err) max_err = err;
        }
    }

    printf("5x5 identity: I^-1 = I, max error = %e %s\n", max_err, max_err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(I);
    mat_free_mat(Iinv);

    return max_err < 1e-5f ? 0 : 1;
}

static int test_inv_random(size_t n) {
    Mat *A = mat_mat(n, n);
    // Make diagonally dominant to ensure invertibility
    for (size_t i = 0; i < n; i++) {
        mat_elem_t row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            mat_elem_t val = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
            A->data[i * n + j] = val;
            if (i != j) row_sum += fabsf(val);
        }
        A->data[i * n + i] = row_sum + 1.0f;
    }

    Mat *Ainv = mat_mat(n, n);
    mat_inv(Ainv, A);

    // Check A * A^-1 = I
    Mat *I = mat_rmul(A, Ainv);

    mat_elem_t max_err = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            mat_elem_t err = fabsf(I->data[i * n + j] - expected);
            if (err > max_err) max_err = err;
        }
    }

    printf("%zux%zu random: A * A^-1 = I, max error = %e %s\n", n, n, max_err, max_err < 1e-4f ? "OK" : "FAIL");

    mat_free_mat(A);
    mat_free_mat(Ainv);
    mat_free_mat(I);

    return max_err < 1e-4f ? 0 : 1;
}

int main(void) {
    srand(42);
    int failures = 0;

    failures += test_inv_2x2();
    failures += test_inv_3x3();
    failures += test_inv_identity();
    failures += test_inv_random(10);
    failures += test_inv_random(50);
    failures += test_inv_random(100);

    printf("\n%s\n", failures == 0 ? "All tests passed!" : "Some tests failed!");
    return failures;
}
