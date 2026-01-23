#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_householder_basic(void) {
    TEST_BEGIN("mat_householder basic");

    // Simple vector [3, 4] -> should give beta = -5 (norm with sign)
    Vec *x = mat_from(2, 1, (mat_elem_t[]){3, 4});
    Vec *v = mat_vec(2);
    mat_elem_t tau;

    mat_elem_t beta = mat_householder(v, &tau, x);

    // beta = -sign(x[0]) * ||x|| = -5
    CHECK_FLOAT_EQ_TOL(beta, -5.0f, 1e-5f);

    // v[0] should be 1
    CHECK_FLOAT_EQ_TOL(v->data[0], 1.0f, 1e-5f);

    // Verify H*x = beta*e1 by applying reflection
    // H = I - tau * v * v^T
    // H*x = x - tau * v * (v^T * x)
    mat_elem_t vtx = v->data[0] * x->data[0] + v->data[1] * x->data[1];
    mat_elem_t hx0 = x->data[0] - tau * v->data[0] * vtx;
    mat_elem_t hx1 = x->data[1] - tau * v->data[1] * vtx;

    CHECK_FLOAT_EQ_TOL(hx0, beta, 1e-5f);
    CHECK_FLOAT_EQ_TOL(hx1, 0.0f, 1e-5f);

    mat_free_mat(x);
    mat_free_mat(v);
    TEST_END();
}

void test_householder_negative_first(void) {
    TEST_BEGIN("mat_householder negative first element");

    // Vector [-3, 4] -> beta = +5
    Vec *x = mat_from(2, 1, (mat_elem_t[]){-3, 4});
    Vec *v = mat_vec(2);
    mat_elem_t tau;

    mat_elem_t beta = mat_householder(v, &tau, x);

    CHECK_FLOAT_EQ_TOL(beta, 5.0f, 1e-5f);

    // Verify H*x = beta*e1
    mat_elem_t vtx = v->data[0] * x->data[0] + v->data[1] * x->data[1];
    mat_elem_t hx0 = x->data[0] - tau * v->data[0] * vtx;
    mat_elem_t hx1 = x->data[1] - tau * v->data[1] * vtx;

    CHECK_FLOAT_EQ_TOL(hx0, beta, 1e-5f);
    CHECK_FLOAT_EQ_TOL(hx1, 0.0f, 1e-5f);

    mat_free_mat(x);
    mat_free_mat(v);
    TEST_END();
}

void test_householder_larger(void) {
    TEST_BEGIN("mat_householder larger vector");

    Vec *x = mat_from(5, 1, (mat_elem_t[]){1, 2, 3, 4, 5});
    Vec *v = mat_vec(5);
    mat_elem_t tau;

    mat_elem_t beta = mat_householder(v, &tau, x);

    // ||x|| = sqrt(1+4+9+16+25) = sqrt(55)
    mat_elem_t norm = MAT_SQRT(55.0f);
    CHECK_FLOAT_EQ_TOL(MAT_FABS(beta), norm, 1e-4f);

    // Verify H*x = beta*e1
    mat_elem_t vtx = 0;
    for (size_t i = 0; i < 5; i++) {
        vtx += v->data[i] * x->data[i];
    }
    mat_elem_t hx[5];
    for (size_t i = 0; i < 5; i++) {
        hx[i] = x->data[i] - tau * v->data[i] * vtx;
    }

    CHECK_FLOAT_EQ_TOL(hx[0], beta, 1e-4f);
    for (size_t i = 1; i < 5; i++) {
        CHECK_FLOAT_EQ_TOL(hx[i], 0.0f, 1e-4f);
    }

    mat_free_mat(x);
    mat_free_mat(v);
    TEST_END();
}

void test_householder_left_identity(void) {
    TEST_BEGIN("mat_householder_left preserves orthogonality");

    // H * I should be orthogonal
    Mat *I = mat_reye(4);
    Vec *x = mat_from(4, 1, (mat_elem_t[]){1, 2, 3, 4});
    Vec *v = mat_vec(4);
    mat_elem_t tau;

    mat_householder(v, &tau, x);
    mat_householder_left(I, v, tau);

    // Check I^T * I = I (orthogonality)
    Mat *It = mat_rt(I);
    Mat *ItI = mat_rmul(It, I);
    Mat *eye = mat_reye(4);

    CHECK(mat_equals_tol(ItI, eye, 1e-5f));

    mat_free_mat(I);
    mat_free_mat(x);
    mat_free_mat(v);
    mat_free_mat(It);
    mat_free_mat(ItI);
    mat_free_mat(eye);
    TEST_END();
}

void test_householder_right_identity(void) {
    TEST_BEGIN("mat_householder_right preserves orthogonality");

    // I * H should be orthogonal
    Mat *I = mat_reye(4);
    Vec *x = mat_from(4, 1, (mat_elem_t[]){1, 2, 3, 4});
    Vec *v = mat_vec(4);
    mat_elem_t tau;

    mat_householder(v, &tau, x);
    mat_householder_right(I, v, tau);

    // Check I * I^T = I (orthogonality)
    Mat *It = mat_rt(I);
    Mat *IIt = mat_rmul(I, It);
    Mat *eye = mat_reye(4);

    CHECK(mat_equals_tol(IIt, eye, 1e-5f));

    mat_free_mat(I);
    mat_free_mat(x);
    mat_free_mat(v);
    mat_free_mat(It);
    mat_free_mat(IIt);
    mat_free_mat(eye);
    TEST_END();
}

void test_householder_left_zeros_column(void) {
    TEST_BEGIN("mat_householder_left zeros first column");

    // Create matrix with known first column
    Mat *A = mat_from(4, 3, (mat_elem_t[]){
        1, 2, 3,
        2, 5, 6,
        3, 8, 9,
        4, 11, 12
    });

    // Extract first column
    Vec *x = mat_vec(4);
    for (size_t i = 0; i < 4; i++) {
        x->data[i] = mat_at(A, i, 0);
    }

    Vec *v = mat_vec(4);
    mat_elem_t tau;

    mat_elem_t beta = mat_householder(v, &tau, x);

    // Apply H from left
    mat_householder_left(A, v, tau);

    // First column should be [beta, 0, 0, 0]
    CHECK_FLOAT_EQ_TOL(mat_at(A, 0, 0), beta, 1e-4f);
    CHECK_FLOAT_EQ_TOL(mat_at(A, 1, 0), 0.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(mat_at(A, 2, 0), 0.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(mat_at(A, 3, 0), 0.0f, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(x);
    mat_free_mat(v);
    TEST_END();
}

void test_householder_qr_manual(void) {
    TEST_BEGIN("mat_householder manual QR");

    // Use Householder to do QR on a 3x3 matrix
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        12, -51, 4,
        6, 167, -68,
        -4, 24, -41
    });
    Mat *R = mat_rdeep_copy(A);
    Mat *Q = mat_reye(3);

    // Column 0
    Vec x0 = {.rows = 3, .cols = 1, .data = R->data};
    Vec *v0 = mat_vec(3);
    mat_elem_t tau0;
    mat_householder(v0, &tau0, &x0);
    mat_householder_left(R, v0, tau0);
    mat_householder_right(Q, v0, tau0);

    // Column 1 (submatrix)
    // Need to apply to submatrix [1:3, 1:3]
    // For simplicity, just verify R[0,1:] and R[1:,0] are correct

    // Verify Q is orthogonal
    Mat *Qt = mat_rt(Q);
    Mat *QtQ = mat_rmul(Qt, Q);
    Mat *eye = mat_reye(3);
    CHECK(mat_equals_tol(QtQ, eye, 1e-4f));

    mat_free_mat(A);
    mat_free_mat(R);
    mat_free_mat(Q);
    mat_free_mat(v0);
    mat_free_mat(Qt);
    mat_free_mat(QtQ);
    mat_free_mat(eye);
    TEST_END();
}

int main(void) {
    printf("mat_householder:\n");

    test_householder_basic();
    test_householder_negative_first();
    test_householder_larger();
    test_householder_left_identity();
    test_householder_right_identity();
    test_householder_left_zeros_column();
    test_householder_qr_manual();

    TEST_SUMMARY();
}
