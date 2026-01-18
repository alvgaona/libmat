#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"
#include <stdlib.h>

// Helper to create a symmetric positive definite matrix: A = B * B^T + n*I
static void make_spd(Mat *A) {
    size_t n = A->rows;

    // Create random matrix B
    Mat *B = mat_mat(n, n);
    for (size_t i = 0; i < n * n; i++) {
        B->data[i] = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
    }

    // A = B * B^T
    Mat *Bt = mat_rt(B);
    Mat *BBt = mat_rmul(B, Bt);
    mat_deep_copy(A, BBt);

    // Add n*I to ensure positive definiteness
    for (size_t i = 0; i < n; i++) {
        A->data[i * n + i] += (mat_elem_t)n;
    }

    mat_free_mat(B);
    mat_free_mat(Bt);
    mat_free_mat(BBt);
}

// Verify L is lower triangular (upper triangle is zero)
static int is_lower_triangular(const Mat *L, mat_elem_t tol) {
    size_t n = L->rows;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (MAT_FABS(L->data[i * n + j]) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

static void test_chol_2x2_identity(void) {
    TEST_BEGIN("chol_2x2_identity");

    Mat *A = mat_mat(2, 2);
    mat_eye(A);
    Mat *L = mat_mat(2, 2);

    int result = mat_chol(A, L);
    CHECK(result == 0);

    // L should be identity for identity matrix
    CHECK_FLOAT_EQ_TOL(L->data[0], 1.0f, 1e-6f);
    CHECK_FLOAT_EQ_TOL(L->data[1], 0.0f, 1e-6f);
    CHECK_FLOAT_EQ_TOL(L->data[2], 0.0f, 1e-6f);
    CHECK_FLOAT_EQ_TOL(L->data[3], 1.0f, 1e-6f);

    mat_free_mat(A);
    mat_free_mat(L);

    TEST_END();
}

static void test_chol_2x2_known(void) {
    TEST_BEGIN("chol_2x2_known");

    // A = [4 2; 2 5] is SPD
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        4, 2,
        2, 5
    });

    Mat *L = mat_mat(2, 2);

    int result = mat_chol(A, L);
    CHECK(result == 0);

    // Expected L = [2 0; 1 2] (L * L^T = A)
    CHECK_FLOAT_EQ_TOL(L->data[0], 2.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(L->data[1], 0.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(L->data[2], 1.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(L->data[3], 2.0f, 1e-5f);

    // Verify L * L^T = A
    Mat *Lt = mat_rt(L);
    Mat *LLt = mat_rmul(L, Lt);

    for (size_t i = 0; i < 4; i++) {
        CHECK_FLOAT_EQ_TOL(A->data[i], LLt->data[i], 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(Lt);
    mat_free_mat(LLt);

    TEST_END();
}

static void test_chol_3x3_known(void) {
    TEST_BEGIN("chol_3x3_known");

    // A = [4 12 -16; 12 37 -43; -16 -43 98] is SPD
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        4, 12, -16,
        12, 37, -43,
        -16, -43, 98
    });

    Mat *L = mat_mat(3, 3);

    int result = mat_chol(A, L);
    CHECK(result == 0);

    // Verify L is lower triangular
    CHECK(is_lower_triangular(L, 1e-5f));

    // Verify L * L^T = A
    Mat *Lt = mat_rt(L);
    Mat *LLt = mat_rmul(L, Lt);

    for (size_t i = 0; i < 9; i++) {
        CHECK_FLOAT_EQ_TOL(A->data[i], LLt->data[i], 1e-4f);
    }

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(Lt);
    mat_free_mat(LLt);

    TEST_END();
}

static void test_chol_not_positive_definite(void) {
    TEST_BEGIN("chol_not_positive_definite");

    // A = [1 2; 2 1] is not positive definite (eigenvalues: 3, -1)
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        1, 2,
        2, 1
    });

    Mat *L = mat_mat(2, 2);

    int result = mat_chol(A, L);
    CHECK(result == -1);

    mat_free_mat(A);
    mat_free_mat(L);

    TEST_END();
}

static void test_chol_random(size_t n, const char *name, mat_elem_t tol) {
    TEST_BEGIN(name);

    Mat *A = mat_mat(n, n);
    make_spd(A);

    Mat *L = mat_mat(n, n);

    int result = mat_chol(A, L);
    CHECK(result == 0);

    // Verify L is lower triangular
    CHECK(is_lower_triangular(L, tol));

    // Verify L * L^T = A
    Mat *Lt = mat_rt(L);
    Mat *LLt = mat_rmul(L, Lt);

    for (size_t i = 0; i < n * n; i++) {
        CHECK_FLOAT_EQ_TOL(A->data[i], LLt->data[i], tol);
    }

    mat_free_mat(A);
    mat_free_mat(L);
    mat_free_mat(Lt);
    mat_free_mat(LLt);

    TEST_END();
}

static void test_chol_diagonal_positive(void) {
    TEST_BEGIN("chol_diagonal_positive");

    // All diagonal elements of L should be positive
    Mat *A = mat_mat(5, 5);
    make_spd(A);

    Mat *L = mat_mat(5, 5);

    int result = mat_chol(A, L);
    CHECK(result == 0);

    for (size_t i = 0; i < 5; i++) {
        CHECK(L->data[i * 5 + i] > 0);
    }

    mat_free_mat(A);
    mat_free_mat(L);

    TEST_END();
}

int main(void) {
    srand(42);

    test_chol_2x2_identity();
    test_chol_2x2_known();
    test_chol_3x3_known();
    test_chol_not_positive_definite();
    test_chol_random(10, "chol_random_10x10", 1e-4f);
    test_chol_random(50, "chol_random_50x50", 1e-3f);
    test_chol_random(100, "chol_random_100x100", 1e-2f);
    test_chol_random(128, "chol_random_128x128", 1e-2f);
    test_chol_random(200, "chol_random_200x200", 5e-2f);
    test_chol_diagonal_positive();

    TEST_SUMMARY();
}
