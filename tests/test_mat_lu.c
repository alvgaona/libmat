#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

static void test_lu_3x3(void) {
    TEST_BEGIN("lu_3x3");

    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        2, -1, -2,
        -4, 6, 3,
        -4, -2, 8
    });

    Mat *L = mat_mat(3, 3);
    Mat *U = mat_mat(3, 3);
    Perm *p = mat_perm(3);
    Perm *q = mat_perm(3);

    mat_lu(A, L, U, p, q);

    // Verify P * A * Q = L * U
    Mat *PAQ = mat_mat(3, 3);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            PAQ->data[i * 3 + j] = A->data[p->data[i] * 3 + q->data[j]];
        }
    }

    Mat *LU = mat_rmul(L, U);

    for (size_t i = 0; i < 9; i++) {
        CHECK_FLOAT_EQ_TOL(LU->data[i], PAQ->data[i], 1e-5f);
    }

    mat_free_mat(A); mat_free_mat(L); mat_free_mat(U);
    mat_free_mat(PAQ); mat_free_mat(LU);
    mat_free_perm(p); mat_free_perm(q);

    TEST_END();
}

static void test_lu_4x4(void) {
    TEST_BEGIN("lu_4x4");

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

    mat_lu(A, L, U, p, q);

    Mat *PAQ = mat_mat(4, 4);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            PAQ->data[i * 4 + j] = A->data[p->data[i] * 4 + q->data[j]];
        }
    }

    Mat *LU = mat_rmul(L, U);

    for (size_t i = 0; i < 16; i++) {
        CHECK_FLOAT_EQ_TOL(LU->data[i], PAQ->data[i], 1e-5f);
    }

    mat_free_mat(A); mat_free_mat(L); mat_free_mat(U);
    mat_free_mat(PAQ); mat_free_mat(LU);
    mat_free_perm(p); mat_free_perm(q);

    TEST_END();
}

static void test_lu_random(size_t n, const char *name) {
    TEST_BEGIN(name);

    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n * n; i++) {
        A->data[i] = (mat_elem_t)(rand() % 200 - 100) / 10.0f;
    }

    Mat *L = mat_mat(n, n);
    Mat *U = mat_mat(n, n);
    Perm *p = mat_perm(n);
    Perm *q = mat_perm(n);

    mat_lu(A, L, U, p, q);

    Mat *PAQ = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            PAQ->data[i * n + j] = A->data[p->data[i] * n + q->data[j]];
        }
    }

    Mat *LU = mat_rmul(L, U);

    for (size_t i = 0; i < n * n; i++) {
        CHECK_FLOAT_EQ_TOL(LU->data[i], PAQ->data[i], 1e-4f);
    }

    mat_free_mat(A); mat_free_mat(L); mat_free_mat(U);
    mat_free_mat(PAQ); mat_free_mat(LU);
    mat_free_perm(p); mat_free_perm(q);

    TEST_END();
}

int main(void) {
    test_lu_3x3();
    test_lu_4x4();
    test_lu_random(10, "lu_random_10x10");
    test_lu_random(50, "lu_random_50x50");
    test_lu_random(100, "lu_random_100x100");

    TEST_SUMMARY();
}
