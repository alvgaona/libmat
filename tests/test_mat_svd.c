#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

// Check if U^T * U = I (U is orthogonal)
static int is_orthogonal_tol(const Mat *U, mat_elem_t tol) {
    Mat *Ut = mat_rt(U);
    Mat *UtU = mat_rmul(Ut, U);
    Mat *I = mat_reye(U->rows);
    int result = mat_equals_tol(UtU, I, tol);
    mat_free_mat(Ut);
    mat_free_mat(UtU);
    mat_free_mat(I);
    return result;
}

static int is_orthogonal(const Mat *U) {
    return is_orthogonal_tol(U, 1e-5f);
}

// Check if Vt * Vt^T = I (rows of Vt are orthonormal)
static int is_orthogonal_rows_tol(const Mat *Vt, mat_elem_t tol) {
    Mat *VtT = mat_rt(Vt);
    Mat *VtVtT = mat_rmul(Vt, VtT);
    Mat *I = mat_reye(Vt->rows);
    int result = mat_equals_tol(VtVtT, I, tol);
    mat_free_mat(VtT);
    mat_free_mat(VtVtT);
    mat_free_mat(I);
    return result;
}

static int is_orthogonal_rows(const Mat *Vt) {
    return is_orthogonal_rows_tol(Vt, 1e-5f);
}

// Check A = U * diag(S) * Vt
static int check_reconstruction(const Mat *A, const Mat *U, const Vec *S, const Mat *Vt, mat_elem_t tol) {
    size_t m = A->rows;
    size_t n = A->cols;
    size_t k = S->rows;

    // Build Sigma (m x n) with S on diagonal
    Mat *Sigma = mat_zeros(m, n);
    for (size_t i = 0; i < k; i++) {
        mat_set_at(Sigma, i, i, S->data[i]);
    }

    // Compute U * Sigma
    Mat *US = mat_rmul(U, Sigma);

    // Compute (U * Sigma) * Vt
    Mat *USVt = mat_rmul(US, Vt);

    int result = mat_equals_tol(A, USVt, tol);

    mat_free_mat(Sigma);
    mat_free_mat(US);
    mat_free_mat(USVt);
    return result;
}

// Check singular values are non-negative and descending
static int check_singular_values(const Vec *S) {
    for (size_t i = 0; i < S->rows; i++) {
        if (S->data[i] < -MAT_DEFAULT_EPSILON) {
            printf("  S[%zu] = %f is negative\n", i, S->data[i]);
            return 0;
        }
    }
    for (size_t i = 0; i < S->rows - 1; i++) {
        if (S->data[i] < S->data[i + 1] - MAT_DEFAULT_EPSILON) {
            printf("  S[%zu] = %f < S[%zu] = %f (not descending)\n",
                   i, S->data[i], i + 1, S->data[i + 1]);
            return 0;
        }
    }
    return 1;
}

void test_svd_2x2_diagonal(void) {
    TEST_BEGIN("mat_svd 2x2 diagonal");
    // Diagonal matrix: SVD should return same diagonal as singular values
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        3, 0,
        0, 2
    });
    Mat *U = mat_mat(2, 2);
    Vec *S = mat_vec(2);
    Mat *Vt = mat_mat(2, 2);

    mat_svd(A, U, S, Vt);

    // Singular values should be 3, 2 (descending)
    CHECK_FLOAT_EQ_TOL(S->data[0], 3.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(S->data[1], 2.0f, 1e-5f);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-5f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_2x2_general(void) {
    TEST_BEGIN("mat_svd 2x2 general");
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        3, 2,
        2, 3
    });
    Mat *U = mat_mat(2, 2);
    Vec *S = mat_vec(2);
    Mat *Vt = mat_mat(2, 2);

    mat_svd(A, U, S, Vt);

    // For symmetric [3,2;2,3], eigenvalues are 5 and 1, so singular values are 5, 1
    CHECK_FLOAT_EQ_TOL(S->data[0], 5.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(S->data[1], 1.0f, 1e-5f);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-5f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_3x3(void) {
    TEST_BEGIN("mat_svd 3x3");
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Mat *U = mat_mat(3, 3);
    Vec *S = mat_vec(3);
    Mat *Vt = mat_mat(3, 3);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-4f));

    // This matrix is rank 2, so third singular value should be ~0
    CHECK(S->data[2] < 1e-5f);

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_4x3(void) {
    TEST_BEGIN("mat_svd 4x3 (tall)");
    Mat *A = mat_from(4, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    });
    Mat *U = mat_mat(4, 4);
    Vec *S = mat_vec(3);  // min(4, 3) = 3
    Mat *Vt = mat_mat(3, 3);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_3x4(void) {
    TEST_BEGIN("mat_svd 3x4 (wide)");
    Mat *A = mat_from(3, 4, (mat_elem_t[]){
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    });
    Mat *U = mat_mat(3, 3);
    Vec *S = mat_vec(3);  // min(3, 4) = 3
    Mat *Vt = mat_mat(4, 4);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_identity(void) {
    TEST_BEGIN("mat_svd identity");
    Mat *A = mat_reye(4);
    Mat *U = mat_mat(4, 4);
    Vec *S = mat_vec(4);
    Mat *Vt = mat_mat(4, 4);

    mat_svd(A, U, S, Vt);

    // All singular values should be 1
    for (size_t i = 0; i < 4; i++) {
        CHECK_FLOAT_EQ_TOL(S->data[i], 1.0f, 1e-5f);
    }

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-5f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_random_10x10(void) {
    TEST_BEGIN("mat_svd 10x10 random");
    size_t n = 10;
    Mat *A = mat_mat(n, n);
    srand(42);
    for (size_t i = 0; i < n * n; i++) {
        A->data[i] = (mat_elem_t)rand() / RAND_MAX * 2 - 1;
    }
    Mat *U = mat_mat(n, n);
    Vec *S = mat_vec(n);
    Mat *Vt = mat_mat(n, n);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_random_50x30(void) {
    TEST_BEGIN("mat_svd 50x30 random (tall)");
    size_t m = 50, n = 30;
    Mat *A = mat_mat(m, n);
    srand(123);
    for (size_t i = 0; i < m * n; i++) {
        A->data[i] = (mat_elem_t)rand() / RAND_MAX * 2 - 1;
    }
    Mat *U = mat_mat(m, m);
    Vec *S = mat_vec(n);  // min(50, 30) = 30
    Mat *Vt = mat_mat(n, n);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_random_30x50(void) {
    TEST_BEGIN("mat_svd 30x50 random (wide)");
    size_t m = 30, n = 50;
    Mat *A = mat_mat(m, n);
    srand(456);
    for (size_t i = 0; i < m * n; i++) {
        A->data[i] = (mat_elem_t)rand() / RAND_MAX * 2 - 1;
    }
    Mat *U = mat_mat(m, m);
    Vec *S = mat_vec(m);  // min(30, 50) = 30
    Mat *Vt = mat_mat(n, n);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal(U));
    CHECK(is_orthogonal_rows(Vt));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

void test_svd_random_100x100(void) {
    TEST_BEGIN("mat_svd 100x100 random");
    size_t n = 100;
    Mat *A = mat_mat(n, n);
    srand(789);
    for (size_t i = 0; i < n * n; i++) {
        A->data[i] = (mat_elem_t)rand() / RAND_MAX * 2 - 1;
    }
    Mat *U = mat_mat(n, n);
    Vec *S = mat_vec(n);
    Mat *Vt = mat_mat(n, n);

    mat_svd(A, U, S, Vt);

    CHECK(is_orthogonal_tol(U, 1e-4f));  // Larger tolerance for 100x100
    CHECK(is_orthogonal_rows_tol(Vt, 1e-4f));
    CHECK(check_singular_values(S));
    CHECK(check_reconstruction(A, U, S, Vt, 1e-3f));

    mat_free_mat(A);
    mat_free_mat(U);
    mat_free_mat(S);
    mat_free_mat(Vt);
    TEST_END();
}

int main(void) {
    printf("mat_svd:\n");

    test_svd_2x2_diagonal();
    test_svd_2x2_general();
    test_svd_3x3();
    test_svd_4x3();
    test_svd_3x4();
    test_svd_identity();
    test_svd_random_10x10();
    test_svd_random_50x30();
    test_svd_random_30x50();
    test_svd_random_100x100();

    TEST_SUMMARY();
}
