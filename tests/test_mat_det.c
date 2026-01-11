#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

static void test_det_2x2(void) {
    TEST_BEGIN("det_2x2");

    // det([[a, b], [c, d]]) = ad - bc
    // det([[3, 8], [4, 6]]) = 3*6 - 8*4 = 18 - 32 = -14
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        3, 8,
        4, 6
    });

    mat_elem_t det = mat_det(A);
    CHECK_FLOAT_EQ_TOL(det, -14.0f, 1e-5f);

    mat_free_mat(A);

    TEST_END();
}

static void test_det_3x3(void) {
    TEST_BEGIN("det_3x3");

    // det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) = -306
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        6, 1, 1,
        4, -2, 5,
        2, 8, 7
    });

    mat_elem_t det = mat_det(A);
    CHECK_FLOAT_EQ_TOL(det, -306.0f, 1e-4f);

    mat_free_mat(A);

    TEST_END();
}

static void test_det_identity(void) {
    TEST_BEGIN("det_identity_5x5");

    // det(I) = 1
    Mat *I = mat_reye(5);

    mat_elem_t det = mat_det(I);
    CHECK_FLOAT_EQ_TOL(det, 1.0f, 1e-5f);

    mat_free_mat(I);

    TEST_END();
}

static void test_det_singular(void) {
    TEST_BEGIN("det_singular_3x3");

    // Singular matrix (row 3 = row 1 + row 2), det = 0
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        5, 7, 9
    });

    mat_elem_t det = mat_det(A);
    CHECK_FLOAT_EQ_TOL(det, 0.0f, 1e-5f);

    mat_free_mat(A);

    TEST_END();
}

static void test_det_4x4(void) {
    TEST_BEGIN("det_4x4");

    // det([[1,2,3,4], [5,6,7,8], [2,6,4,8], [3,1,1,2]]) = 72
    Mat *A = mat_from(4, 4, (mat_elem_t[]){
        1, 2, 3, 4,
        5, 6, 7, 8,
        2, 6, 4, 8,
        3, 1, 1, 2
    });

    mat_elem_t det = mat_det(A);
    CHECK_FLOAT_EQ_TOL(det, 72.0f, 1e-4f);

    mat_free_mat(A);

    TEST_END();
}

int main(void) {
    test_det_2x2();
    test_det_3x3();
    test_det_identity();
    test_det_singular();
    test_det_4x4();

    TEST_SUMMARY();
}
