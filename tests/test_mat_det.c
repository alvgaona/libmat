#define MAT_IMPLEMENTATION
#include "mat.h"
#include <stdio.h>

static int test_det_2x2(void) {
    // det([[a, b], [c, d]]) = ad - bc
    // det([[3, 8], [4, 6]]) = 3*6 - 8*4 = 18 - 32 = -14
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        3, 8,
        4, 6
    });

    mat_elem_t det = mat_det(A);
    mat_elem_t expected = -14.0f;
    mat_elem_t err = fabsf(det - expected);

    printf("2x2: det = %.4f, expected = %.4f, error = %e %s\n",
           det, expected, err, err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A);
    return err < 1e-5f ? 0 : 1;
}

static int test_det_3x3(void) {
    // det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) = -306
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        6, 1, 1,
        4, -2, 5,
        2, 8, 7
    });

    mat_elem_t det = mat_det(A);
    mat_elem_t expected = -306.0f;
    mat_elem_t err = fabsf(det - expected);

    printf("3x3: det = %.4f, expected = %.4f, error = %e %s\n",
           det, expected, err, err < 1e-4f ? "OK" : "FAIL");

    mat_free_mat(A);
    return err < 1e-4f ? 0 : 1;
}

static int test_det_identity(void) {
    // det(I) = 1
    Mat *I = mat_reye(5);

    mat_elem_t det = mat_det(I);
    mat_elem_t expected = 1.0f;
    mat_elem_t err = fabsf(det - expected);

    printf("5x5 identity: det = %.4f, expected = %.4f, error = %e %s\n",
           det, expected, err, err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(I);
    return err < 1e-5f ? 0 : 1;
}

static int test_det_singular(void) {
    // Singular matrix (row 3 = row 1 + row 2), det = 0
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        5, 7, 9
    });

    mat_elem_t det = mat_det(A);
    mat_elem_t err = fabsf(det);

    printf("3x3 singular: det = %.4f, expected = 0, error = %e %s\n",
           det, err, err < 1e-5f ? "OK" : "FAIL");

    mat_free_mat(A);
    return err < 1e-5f ? 0 : 1;
}

static int test_det_4x4(void) {
    // det([[1,2,3,4], [5,6,7,8], [2,6,4,8], [3,1,1,2]]) = 72
    Mat *A = mat_from(4, 4, (mat_elem_t[]){
        1, 2, 3, 4,
        5, 6, 7, 8,
        2, 6, 4, 8,
        3, 1, 1, 2
    });

    mat_elem_t det = mat_det(A);
    mat_elem_t expected = 72.0f;
    mat_elem_t err = fabsf(det - expected);

    printf("4x4: det = %.4f, expected = %.4f, error = %e %s\n",
           det, expected, err, err < 1e-4f ? "OK" : "FAIL");

    mat_free_mat(A);
    return err < 1e-4f ? 0 : 1;
}

int main(void) {
    int failures = 0;

    failures += test_det_2x2();
    failures += test_det_3x3();
    failures += test_det_identity();
    failures += test_det_singular();
    failures += test_det_4x4();

    printf("\n%s\n", failures == 0 ? "All tests passed!" : "Some tests failed!");
    return failures;
}
