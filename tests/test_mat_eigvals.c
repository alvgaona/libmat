#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

// Sort eigenvalues in descending order for comparison
static void sort_desc(mat_elem_t *arr, size_t n) {
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (arr[j] > arr[i]) {
                mat_elem_t tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }
}

void test_eigvals_1x1(void) {
    TEST_BEGIN("mat_eigvals 1x1");
    Mat *A = mat_from(1, 1, (mat_elem_t[]){7.0f});
    Vec *eig = mat_vec(1);

    mat_eigvals(eig, A);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 7.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_2x2_diagonal(void) {
    TEST_BEGIN("mat_eigvals 2x2 diagonal");
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        5, 0,
        0, 3
    });
    Vec *eig = mat_vec(2);

    mat_eigvals(eig, A);
    sort_desc(eig->data, 2);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 5.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 3.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_2x2_symmetric(void) {
    TEST_BEGIN("mat_eigvals 2x2 symmetric");
    // [[2,1],[1,2]] has eigenvalues 3, 1
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        2, 1,
        1, 2
    });
    Vec *eig = mat_vec(2);

    mat_eigvals(eig, A);
    sort_desc(eig->data, 2);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 3.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 1.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_3x3_identity(void) {
    TEST_BEGIN("mat_eigvals 3x3 identity");
    Mat *A = mat_reye(3);
    Vec *eig = mat_vec(3);

    mat_eigvals(eig, A);

    for (size_t i = 0; i < 3; i++) {
        CHECK_FLOAT_EQ_TOL(eig->data[i], 1.0f, 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_3x3_symmetric(void) {
    TEST_BEGIN("mat_eigvals 3x3 symmetric");
    // [[3,1,1],[1,3,1],[1,1,3]] has eigenvalues 5, 2, 2
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        3, 1, 1,
        1, 3, 1,
        1, 1, 3
    });
    Vec *eig = mat_vec(3);

    mat_eigvals(eig, A);
    sort_desc(eig->data, 3);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 5.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 2.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[2], 2.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_4x4_diagonal(void) {
    TEST_BEGIN("mat_eigvals 4x4 diagonal");
    Mat *A = mat_from(4, 4, (mat_elem_t[]){
        4, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 1
    });
    Vec *eig = mat_vec(4);

    mat_eigvals(eig, A);
    sort_desc(eig->data, 4);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 4.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 3.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[2], 2.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[3], 1.0f, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_5x5_trace(void) {
    TEST_BEGIN("mat_eigvals 5x5 trace check");
    // For any matrix, sum of eigenvalues = trace
    srand(42);
    Mat *A = mat_mat(5, 5);
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = i; j < 5; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    mat_elem_t trace = mat_trace(A);
    Vec *eig = mat_vec(5);
    mat_eigvals(eig, A);

    mat_elem_t eig_sum = 0;
    for (size_t i = 0; i < 5; i++) {
        eig_sum += eig->data[i];
    }

    CHECK_FLOAT_EQ_TOL(eig_sum, trace, 1e-3f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_10x10_trace(void) {
    TEST_BEGIN("mat_eigvals 10x10 trace check");
    srand(123);
    size_t n = 10;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    mat_elem_t trace = mat_trace(A);
    Vec *eig = mat_vec(n);
    mat_eigvals(eig, A);

    mat_elem_t eig_sum = 0;
    for (size_t i = 0; i < n; i++) {
        eig_sum += eig->data[i];
    }

    CHECK_FLOAT_EQ_TOL(eig_sum, trace, 1e-2f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_non_symmetric(void) {
    TEST_BEGIN("mat_eigvals 3x3 non-symmetric trace check");
    // For any matrix, sum of eigenvalues = trace
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        2, 0, 0,
        1, 2, 0,
        0, 1, 3
    });
    Vec *eig = mat_vec(3);

    mat_eigvals(eig, A);

    mat_elem_t trace = mat_trace(A);
    mat_elem_t eig_sum = eig->data[0] + eig->data[1] + eig->data[2];

    CHECK_FLOAT_EQ_TOL(eig_sum, trace, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

/* ========================================================================== */
/* mat_eigvals_sym tests                                                      */
/* ========================================================================== */

void test_eigvals_sym_1x1(void) {
    TEST_BEGIN("mat_eigvals_sym 1x1");
    Mat *A = mat_from(1, 1, (mat_elem_t[]){7.0f});
    Vec *eig = mat_vec(1);

    mat_eigvals_sym(eig, A);
    CHECK_FLOAT_EQ_TOL(eig->data[0], 7.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_sym_2x2(void) {
    TEST_BEGIN("mat_eigvals_sym 2x2");
    // [[2,1],[1,2]] has eigenvalues 3, 1
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        2, 1,
        1, 2
    });
    Vec *eig = mat_vec(2);

    mat_eigvals_sym(eig, A);
    sort_desc(eig->data, 2);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 3.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 1.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_sym_3x3(void) {
    TEST_BEGIN("mat_eigvals_sym 3x3");
    // [[3,1,1],[1,3,1],[1,1,3]] has eigenvalues 5, 2, 2
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        3, 1, 1,
        1, 3, 1,
        1, 1, 3
    });
    Vec *eig = mat_vec(3);

    mat_eigvals_sym(eig, A);
    sort_desc(eig->data, 3);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 5.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 2.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[2], 2.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_sym_identity(void) {
    TEST_BEGIN("mat_eigvals_sym identity");
    Mat *A = mat_reye(5);
    Vec *eig = mat_vec(5);

    mat_eigvals_sym(eig, A);

    for (size_t i = 0; i < 5; i++) {
        CHECK_FLOAT_EQ_TOL(eig->data[i], 1.0f, 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_sym_diagonal(void) {
    TEST_BEGIN("mat_eigvals_sym diagonal");
    Mat *A = mat_from(4, 4, (mat_elem_t[]){
        4, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 1
    });
    Vec *eig = mat_vec(4);

    mat_eigvals_sym(eig, A);
    sort_desc(eig->data, 4);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 4.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 3.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[2], 2.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[3], 1.0f, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_sym_trace_10x10(void) {
    TEST_BEGIN("mat_eigvals_sym 10x10 trace check");
    srand(123);
    size_t n = 10;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    mat_elem_t trace = mat_trace(A);
    Vec *eig = mat_vec(n);
    mat_eigvals_sym(eig, A);

    mat_elem_t eig_sum = 0;
    for (size_t i = 0; i < n; i++) {
        eig_sum += eig->data[i];
    }

    CHECK_FLOAT_EQ_TOL(eig_sum, trace, 1e-2f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigvals_sym_compare_with_general(void) {
    TEST_BEGIN("mat_eigvals_sym matches mat_eigvals");
    srand(456);
    size_t n = 8;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    Vec *eig_sym = mat_vec(n);
    Vec *eig_gen = mat_vec(n);

    mat_eigvals_sym(eig_sym, A);
    mat_eigvals(eig_gen, A);

    sort_desc(eig_sym->data, n);
    sort_desc(eig_gen->data, n);

    for (size_t i = 0; i < n; i++) {
        CHECK_FLOAT_EQ_TOL(eig_sym->data[i], eig_gen->data[i], 1e-3f);
    }

    mat_free_mat(A);
    mat_free_mat(eig_sym);
    mat_free_mat(eig_gen);
    TEST_END();
}

void test_eigvals_sym_50x50_trace(void) {
    TEST_BEGIN("mat_eigvals_sym 50x50 trace check");
    srand(789);
    size_t n = 50;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    mat_elem_t trace = mat_trace(A);
    Vec *eig = mat_vec(n);
    mat_eigvals_sym(eig, A);

    mat_elem_t eig_sum = 0;
    for (size_t i = 0; i < n; i++) {
        eig_sum += eig->data[i];
    }

    CHECK_FLOAT_EQ_TOL(eig_sum, trace, 0.5f);

    mat_free_mat(A);
    mat_free_mat(eig);
    TEST_END();
}

int main(void) {
    printf("mat_eigvals:\n");

    test_eigvals_1x1();
    test_eigvals_2x2_diagonal();
    test_eigvals_2x2_symmetric();
    test_eigvals_3x3_identity();
    test_eigvals_3x3_symmetric();
    test_eigvals_4x4_diagonal();
    test_eigvals_5x5_trace();
    test_eigvals_10x10_trace();
    test_eigvals_non_symmetric();

    printf("\nmat_eigvals_sym:\n");

    test_eigvals_sym_1x1();
    test_eigvals_sym_2x2();
    test_eigvals_sym_3x3();
    test_eigvals_sym_identity();
    test_eigvals_sym_diagonal();
    test_eigvals_sym_trace_10x10();
    test_eigvals_sym_compare_with_general();
    test_eigvals_sym_50x50_trace();

    TEST_SUMMARY();
}
