#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

// Check if matrix is upper triangular (below diagonal is zero)
// Tolerance scales with matrix dimension to account for float32 precision
static int is_upper_triangular(const Mat *R) {
    mat_elem_t tol = (mat_elem_t)R->rows * MAT_DEFAULT_EPSILON;
    for (size_t i = 1; i < R->rows; i++) {
        for (size_t j = 0; j < i && j < R->cols; j++) {
            if (MAT_FABS(MAT_AT(R, i, j)) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

// Check if Q^T * Q = I (Q is orthogonal)
static int is_orthogonal(const Mat *Q) {
    Mat *Qt = mat_rt(Q);
    Mat *QtQ = mat_rmul(Qt, Q);
    Mat *I = mat_reye(Q->rows);
    int result = mat_equals_tol(QtQ, I, 1e-5f);
    mat_free_mat(Qt);
    mat_free_mat(QtQ);
    mat_free_mat(I);
    return result;
}

void test_qr_3x3(void) {
    TEST_BEGIN("mat_qr 3x3");
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        12, -51,   4,
         6, 167, -68,
        -4,  24, -41
    });
    Mat *Q = mat_mat(3, 3);
    Mat *R = mat_mat(3, 3);

    mat_qr(A, Q, R);

    // Check Q is orthogonal
    CHECK(is_orthogonal(Q));

    // Check R is upper triangular
    CHECK(is_upper_triangular(R));

    // Check A = Q * R (larger tolerance for large values in float32)
    Mat *QR = mat_rmul(Q, R);
    CHECK(mat_equals_tol(A, QR, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(Q);
    mat_free_mat(R);
    mat_free_mat(QR);
    TEST_END();
}

void test_qr_4x3(void) {
    TEST_BEGIN("mat_qr 4x3 (overdetermined)");
    Mat *A = mat_from(4, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    });
    Mat *Q = mat_mat(4, 4);
    Mat *R = mat_mat(4, 3);

    mat_qr(A, Q, R);

    CHECK(is_orthogonal(Q));
    CHECK(is_upper_triangular(R));

    Mat *QR = mat_rmul(Q, R);
    CHECK(mat_equals_tol(A, QR, 1e-5f));

    mat_free_mat(A);
    mat_free_mat(Q);
    mat_free_mat(R);
    mat_free_mat(QR);
    TEST_END();
}

void test_qr_identity(void) {
    TEST_BEGIN("mat_qr identity matrix");
    Mat *A = mat_reye(3);
    Mat *Q = mat_mat(3, 3);
    Mat *R = mat_mat(3, 3);

    mat_qr(A, Q, R);

    CHECK(is_orthogonal(Q));
    CHECK(is_upper_triangular(R));

    Mat *QR = mat_rmul(Q, R);
    CHECK(mat_equals_tol(A, QR, 1e-5f));

    mat_free_mat(A);
    mat_free_mat(Q);
    mat_free_mat(R);
    mat_free_mat(QR);
    TEST_END();
}

void test_qr_random_100x50(void) {
    TEST_BEGIN("mat_qr 100x50 random");
    size_t m = 100, n = 50;
    Mat *A = mat_mat(m, n);
    srand(42);
    for (size_t i = 0; i < m * n; i++) {
        A->data[i] = (mat_elem_t)rand() / RAND_MAX * 2 - 1;
    }
    Mat *Q = mat_mat(m, m);
    Mat *R = mat_mat(m, n);

    mat_qr(A, Q, R);

    CHECK(is_orthogonal(Q));
    CHECK(is_upper_triangular(R));

    Mat *QR = mat_rmul(Q, R);
    CHECK(mat_equals_tol(A, QR, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(Q);
    mat_free_mat(R);
    mat_free_mat(QR);
    TEST_END();
}

int main(void) {
    printf("mat_qr:\n");

    test_qr_3x3();
    test_qr_4x3();
    test_qr_identity();
    test_qr_random_100x50();

    TEST_SUMMARY();
}
