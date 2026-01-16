#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_rank_full_rank(void) {
    TEST_BEGIN("mat_rank full rank matrix");

    // Identity matrix: rank = n
    Mat *I = mat_reye(4);
    CHECK(mat_rank(I) == 4);
    mat_free_mat(I);

    // Random full-rank 3x3
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 7,  // 7 instead of 6 to avoid singularity
        7, 8, 10
    });
    CHECK(mat_rank(A) == 3);
    mat_free_mat(A);

    TEST_END();
}

void test_rank_deficient(void) {
    TEST_BEGIN("mat_rank rank-deficient matrix");

    // Rank 2 matrix (rows 1,2,3 are linear combinations)
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    CHECK(mat_rank(A) == 2);
    mat_free_mat(A);

    // Rank 1 matrix (all rows are multiples)
    Mat *B = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        2, 4, 6,
        3, 6, 9
    });
    CHECK(mat_rank(B) == 1);
    mat_free_mat(B);

    // Zero matrix: rank = 0
    Mat *Z = mat_zeros(3, 3);
    CHECK(mat_rank(Z) == 0);
    mat_free_mat(Z);

    TEST_END();
}

void test_rank_rectangular(void) {
    TEST_BEGIN("mat_rank rectangular matrices");

    // Tall matrix 4x3, full rank
    Mat *A = mat_from(4, 3, (mat_elem_t[]){
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 1, 1
    });
    CHECK(mat_rank(A) == 3);
    mat_free_mat(A);

    // Wide matrix 3x4, full rank
    Mat *B = mat_from(3, 4, (mat_elem_t[]){
        1, 0, 0, 1,
        0, 2, 0, 1,
        0, 0, 3, 1
    });
    CHECK(mat_rank(B) == 3);
    mat_free_mat(B);

    // Tall matrix 4x3, rank 2
    Mat *C = mat_from(4, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    });
    CHECK(mat_rank(C) == 2);
    mat_free_mat(C);

    TEST_END();
}

void test_cond_identity(void) {
    TEST_BEGIN("mat_cond identity matrix");

    // Identity: condition number = 1
    Mat *I = mat_reye(5);
    mat_elem_t cond = mat_cond(I);
    CHECK_FLOAT_EQ_TOL(cond, 1.0f, 1e-5f);
    mat_free_mat(I);

    TEST_END();
}

void test_cond_diagonal(void) {
    TEST_BEGIN("mat_cond diagonal matrix");

    // Diagonal with values 10, 5, 1: cond = 10/1 = 10
    Mat *D = mat_from(3, 3, (mat_elem_t[]){
        10, 0, 0,
        0, 5, 0,
        0, 0, 1
    });
    mat_elem_t cond = mat_cond(D);
    CHECK_FLOAT_EQ_TOL(cond, 10.0f, 1e-4f);
    mat_free_mat(D);

    TEST_END();
}

void test_cond_well_conditioned(void) {
    TEST_BEGIN("mat_cond well-conditioned matrix");

    // Symmetric positive definite: should have small condition number
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        4, 1, 1,
        1, 4, 1,
        1, 1, 4
    });
    mat_elem_t cond = mat_cond(A);
    // Eigenvalues are 2, 5, 5, so singular values same, cond = 5/2 = 2.5
    CHECK(cond < 10);
    CHECK(cond > 1);
    mat_free_mat(A);

    TEST_END();
}

void test_cond_ill_conditioned(void) {
    TEST_BEGIN("mat_cond ill-conditioned matrix");

    // Near-singular: very large condition number
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9.00001f  // Almost singular
    });
    mat_elem_t cond = mat_cond(A);
    CHECK(cond > 1e4);  // Very large
    mat_free_mat(A);

    TEST_END();
}

void test_cond_singular(void) {
    TEST_BEGIN("mat_cond singular matrix");

    // Singular matrix: cond = infinity
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    mat_elem_t cond = mat_cond(A);
    CHECK(cond == MAT_HUGE_VAL);
    mat_free_mat(A);

    TEST_END();
}

void test_cond_rectangular(void) {
    TEST_BEGIN("mat_cond rectangular matrices");

    // Tall matrix 4x2
    Mat *A = mat_from(4, 2, (mat_elem_t[]){
        1, 0,
        0, 2,
        0, 0,
        0, 0
    });
    mat_elem_t cond = mat_cond(A);
    CHECK_FLOAT_EQ_TOL(cond, 2.0f, 1e-4f);
    mat_free_mat(A);

    // Wide matrix 2x4
    Mat *B = mat_from(2, 4, (mat_elem_t[]){
        3, 0, 0, 0,
        0, 1, 0, 0
    });
    mat_elem_t cond_b = mat_cond(B);
    CHECK_FLOAT_EQ_TOL(cond_b, 3.0f, 1e-4f);
    mat_free_mat(B);

    TEST_END();
}

int main(void) {
    printf("mat_rank and mat_cond:\n");

    test_rank_full_rank();
    test_rank_deficient();
    test_rank_rectangular();
    test_cond_identity();
    test_cond_diagonal();
    test_cond_well_conditioned();
    test_cond_ill_conditioned();
    test_cond_singular();
    test_cond_rectangular();

    TEST_SUMMARY();
}
