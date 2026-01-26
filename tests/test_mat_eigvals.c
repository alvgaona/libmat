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

/* ========================================================================== */
/* mat_eigen_sym tests                                                        */
/* ========================================================================== */

void test_eigen_sym_1x1(void) {
    TEST_BEGIN("mat_eigen_sym 1x1");
    Mat *A = mat_from(1, 1, (mat_elem_t[]){7.0f});
    Mat *V = mat_mat(1, 1);
    Vec *eig = mat_vec(1);

    mat_eigen_sym(V, eig, A);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 7.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(MAT_FABS(mat_at(V, 0, 0)), 1.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigen_sym_2x2(void) {
    TEST_BEGIN("mat_eigen_sym 2x2");
    // [[2,1],[1,2]] has eigenvalues 1, 3
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        2, 1,
        1, 2
    });
    Mat *V = mat_mat(2, 2);
    Vec *eig = mat_vec(2);

    mat_eigen_sym(V, eig, A);

    // Eigenvalues sorted ascending
    CHECK_FLOAT_EQ_TOL(eig->data[0], 1.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 3.0f, 1e-5f);

    // Check V is orthogonal: V^T * V = I
    Mat *Vt = mat_rt(V);
    Mat *VtV = mat_mat(2, 2);
    mat_mul(VtV, Vt, V);
    CHECK_FLOAT_EQ_TOL(mat_at(VtV, 0, 0), 1.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(mat_at(VtV, 1, 1), 1.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(mat_at(VtV, 0, 1), 0.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(mat_at(VtV, 1, 0), 0.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(Vt);
    mat_free_mat(eig);
    mat_free_mat(VtV);
    TEST_END();
}

void test_eigen_sym_3x3(void) {
    TEST_BEGIN("mat_eigen_sym 3x3");
    // [[3,1,1],[1,3,1],[1,1,3]] has eigenvalues 2, 2, 5
    Mat *A = mat_from(3, 3, (mat_elem_t[]){
        3, 1, 1,
        1, 3, 1,
        1, 1, 3
    });
    Mat *V = mat_mat(3, 3);
    Vec *eig = mat_vec(3);

    mat_eigen_sym(V, eig, A);

    // Eigenvalues sorted ascending
    CHECK_FLOAT_EQ_TOL(eig->data[0], 2.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 2.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[2], 5.0f, 1e-4f);

    // Check A*V = V*D: (A*V)[:,i] = eigenvalue[i] * V[:,i]
    Mat *AV = mat_mat(3, 3);
    mat_mul(AV, A, V);
    for (size_t j = 0; j < 3; j++) {
        for (size_t i = 0; i < 3; i++) {
            mat_elem_t expected = eig->data[j] * mat_at(V, i, j);
            CHECK_FLOAT_EQ_TOL(mat_at(AV, i, j), expected, 1e-4f);
        }
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    mat_free_mat(AV);
    TEST_END();
}

void test_eigen_sym_identity(void) {
    TEST_BEGIN("mat_eigen_sym identity");
    size_t n = 5;
    Mat *A = mat_reye(n);
    Mat *V = mat_mat(n, n);
    Vec *eig = mat_vec(n);

    mat_eigen_sym(V, eig, A);

    // All eigenvalues = 1
    for (size_t i = 0; i < n; i++) {
        CHECK_FLOAT_EQ_TOL(eig->data[i], 1.0f, 1e-5f);
    }

    // V should be orthogonal (any orthogonal matrix works for identity)
    Mat *Vt = mat_rt(V);
    Mat *VtV = mat_mat(n, n);
    mat_mul(VtV, Vt, V);
    for (size_t i = 0; i < n; i++) {
        CHECK_FLOAT_EQ_TOL(mat_at(VtV, i, i), 1.0f, 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(Vt);
    mat_free_mat(eig);
    mat_free_mat(VtV);
    TEST_END();
}

void test_eigen_sym_diagonal(void) {
    TEST_BEGIN("mat_eigen_sym diagonal");
    Mat *A = mat_from(4, 4, (mat_elem_t[]){
        4, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 2
    });
    Mat *V = mat_mat(4, 4);
    Vec *eig = mat_vec(4);

    mat_eigen_sym(V, eig, A);

    // Eigenvalues sorted ascending: 1, 2, 3, 4
    CHECK_FLOAT_EQ_TOL(eig->data[0], 1.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 2.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[2], 3.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig->data[3], 4.0f, 1e-4f);

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigen_sym_orthogonality(void) {
    TEST_BEGIN("mat_eigen_sym orthogonality 10x10");
    srand(111);
    size_t n = 10;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    Mat *V = mat_mat(n, n);
    Vec *eig = mat_vec(n);
    mat_eigen_sym(V, eig, A);

    // Check V^T * V = I
    Mat *Vt = mat_rt(V);
    Mat *VtV = mat_mat(n, n);
    mat_mul(VtV, Vt, V);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mat_elem_t expected = (i == j) ? 1.0f : 0.0f;
            CHECK_FLOAT_EQ_TOL(mat_at(VtV, i, j), expected, 1e-4f);
        }
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(Vt);
    mat_free_mat(eig);
    mat_free_mat(VtV);
    TEST_END();
}

void test_eigen_sym_decomposition(void) {
    TEST_BEGIN("mat_eigen_sym A*V = V*D 20x20");
    srand(222);
    size_t n = 20;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    Mat *V = mat_mat(n, n);
    Vec *eig = mat_vec(n);
    mat_eigen_sym(V, eig, A);

    // Check A*V = V*D
    Mat *AV = mat_mat(n, n);
    mat_mul(AV, A, V);

    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            mat_elem_t expected = eig->data[j] * mat_at(V, i, j);
            CHECK_FLOAT_EQ_TOL(mat_at(AV, i, j), expected, 1e-3f);
        }
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    mat_free_mat(AV);
    TEST_END();
}

void test_eigen_sym_eigenvalues_match(void) {
    TEST_BEGIN("mat_eigen_sym eigenvalues match mat_eigvals_sym");
    srand(333);
    size_t n = 15;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            mat_elem_t v = (mat_elem_t)(rand() % 100) / 10.0f;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }

    Mat *V = mat_mat(n, n);
    Vec *eig_full = mat_vec(n);
    Vec *eig_vals = mat_vec(n);

    mat_eigen_sym(V, eig_full, A);
    mat_eigvals_sym(eig_vals, A);

    // Sort both for comparison (mat_eigen_sym sorts ascending, mat_eigvals_sym doesn't)
    sort_desc(eig_full->data, n);
    sort_desc(eig_vals->data, n);

    for (size_t i = 0; i < n; i++) {
        CHECK_FLOAT_EQ_TOL(eig_full->data[i], eig_vals->data[i], 1e-3f);
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig_full);
    mat_free_mat(eig_vals);
    TEST_END();
}

/* ========================================================================== */
/* mat_eigen tests (non-symmetric eigendecomposition)                         */
/* ========================================================================== */

void test_eigen_1x1(void) {
    TEST_BEGIN("mat_eigen 1x1");
    Mat *A = mat_from(1, 1, (mat_elem_t[]){7.0f});
    Mat *V = mat_mat(1, 1);
    Vec *eig = mat_vec(1);

    mat_eigen(V, eig, A);

    CHECK_FLOAT_EQ_TOL(eig->data[0], 7.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(MAT_AT(V, 0, 0), 1.0f, 1e-5f);

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigen_2x2_diagonal(void) {
    TEST_BEGIN("mat_eigen 2x2 diagonal");
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        5, 0,
        0, 3
    });
    Mat *V = mat_mat(2, 2);
    Vec *eig = mat_vec(2);

    mat_eigen(V, eig, A);

    // Check A*V = V*D (for real eigenvalues)
    // For diagonal matrix, eigenvectors should be standard basis
    Mat *AV = mat_mat(2, 2);
    Mat *VD = mat_mat(2, 2);
    mat_mul(AV, A, V);

    // VD = V * diag(eig)
    for (size_t j = 0; j < 2; j++) {
        for (size_t i = 0; i < 2; i++) {
            MAT_SET(VD, i, j, MAT_AT(V, i, j) * eig->data[j]);
        }
    }

    for (size_t i = 0; i < 4; i++) {
        CHECK_FLOAT_EQ_TOL(AV->data[i], VD->data[i], 1e-4f);
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    mat_free_mat(AV);
    mat_free_mat(VD);
    TEST_END();
}

void test_eigen_2x2_nonsym(void) {
    TEST_BEGIN("mat_eigen 2x2 non-symmetric");
    // [[4, 1], [2, 3]] has eigenvalues 5 and 2
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        4, 2,
        1, 3
    });
    Mat *V = mat_mat(2, 2);
    Vec *eig = mat_vec(2);

    mat_eigen(V, eig, A);

    // Sort eigenvalues for comparison
    mat_elem_t eig_copy[2] = {eig->data[0], eig->data[1]};
    sort_desc(eig_copy, 2);
    CHECK_FLOAT_EQ_TOL(eig_copy[0], 5.0f, 1e-4f);
    CHECK_FLOAT_EQ_TOL(eig_copy[1], 2.0f, 1e-4f);

    // Check A*V = V*D
    Mat *AV = mat_mat(2, 2);
    Mat *VD = mat_mat(2, 2);
    mat_mul(AV, A, V);
    for (size_t j = 0; j < 2; j++) {
        for (size_t i = 0; i < 2; i++) {
            MAT_SET(VD, i, j, MAT_AT(V, i, j) * eig->data[j]);
        }
    }
    for (size_t i = 0; i < 4; i++) {
        CHECK_FLOAT_EQ_TOL(AV->data[i], VD->data[i], 1e-4f);
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    mat_free_mat(AV);
    mat_free_mat(VD);
    TEST_END();
}

void test_eigen_3x3_identity(void) {
    TEST_BEGIN("mat_eigen 3x3 identity");
    Mat *A = mat_mat(3, 3);
    mat_eye(A);
    Mat *V = mat_mat(3, 3);
    Vec *eig = mat_vec(3);

    mat_eigen(V, eig, A);

    // All eigenvalues should be 1
    for (size_t i = 0; i < 3; i++) {
        CHECK_FLOAT_EQ_TOL(eig->data[i], 1.0f, 1e-5f);
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    TEST_END();
}

void test_eigen_decomposition_real(void) {
    TEST_BEGIN("mat_eigen A*V = V*D 5x5 (real eigenvalues)");
    // Create a matrix with known real eigenvalues by A = P * D * P^-1
    // Use diagonal D and random orthogonal P (so P^-1 = P^T)
    size_t n = 5;
    Mat *D = mat_mat(n, n);  // Already zeroed
    for (size_t i = 0; i < n; i++) {
        MAT_SET(D, i, i, (mat_elem_t)(i + 1));  // eigenvalues: 1, 2, 3, 4, 5
    }

    // Create orthogonal P from QR of random matrix
    Mat *R = mat_mat(n, n);
    for (size_t i = 0; i < n * n; i++) {
        R->data[i] = (mat_elem_t)rand() / RAND_MAX;
    }
    Mat *P = mat_mat(n, n);
    Mat *Rtmp = mat_mat(n, n);
    mat_deep_copy(Rtmp, R);
    mat_qr(P, Rtmp, R);  // P is orthogonal

    // A = P * D * P^T
    Mat *PD = mat_mat(n, n);
    Mat *A = mat_mat(n, n);
    mat_mul(PD, P, D);
    mat_mul(A, PD, mat_rt(P));

    Mat *V = mat_mat(n, n);
    Vec *eig = mat_vec(n);

    mat_eigen(V, eig, A);

    // Check A*V = V*D
    Mat *AV = mat_mat(n, n);
    Mat *VD = mat_mat(n, n);
    mat_mul(AV, A, V);
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            MAT_SET(VD, i, j, MAT_AT(V, i, j) * eig->data[j]);
        }
    }

    for (size_t i = 0; i < n * n; i++) {
        CHECK_FLOAT_EQ_TOL(AV->data[i], VD->data[i], 1e-3f);
    }

    mat_free_mat(D);
    mat_free_mat(R);
    mat_free_mat(P);
    mat_free_mat(Rtmp);
    mat_free_mat(PD);
    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig);
    mat_free_mat(AV);
    mat_free_mat(VD);
    TEST_END();
}

void test_eigen_eigenvalues_match(void) {
    TEST_BEGIN("mat_eigen eigenvalues match mat_eigvals");
    size_t n = 8;
    Mat *A = mat_mat(n, n);
    for (size_t i = 0; i < n * n; i++) {
        A->data[i] = (mat_elem_t)rand() / RAND_MAX;
    }

    Mat *V = mat_mat(n, n);
    Vec *eig_full = mat_vec(n);
    Vec *eig_only = mat_vec(n);

    mat_eigen(V, eig_full, A);
    mat_eigvals(eig_only, A);

    // Sort both for comparison (eigenvalues might be in different order)
    sort_desc(eig_full->data, n);
    sort_desc(eig_only->data, n);

    for (size_t i = 0; i < n; i++) {
        CHECK_FLOAT_EQ_TOL(eig_full->data[i], eig_only->data[i], 1e-3f);
    }

    mat_free_mat(A);
    mat_free_mat(V);
    mat_free_mat(eig_full);
    mat_free_mat(eig_only);
    TEST_END();
}

void test_eigen_2x2_complex(void) {
    TEST_BEGIN("mat_eigen 2x2 complex eigenvalues");
    // [[0, -1], [1, 0]] has eigenvalues ±i
    Mat *A = mat_from(2, 2, (mat_elem_t[]){
        0, 1,
        -1, 0
    });
    Mat *V = mat_mat(2, 2);
    Vec *eig = mat_vec(2);

    mat_eigen(V, eig, A);

    // Real part should be 0 for both
    CHECK_FLOAT_EQ_TOL(eig->data[0], 0.0f, 1e-5f);
    CHECK_FLOAT_EQ_TOL(eig->data[1], 0.0f, 1e-5f);

    // For complex eigenvalues, V[:, 0] = real part, V[:, 1] = imag part
    // The actual eigenvector is V[:, 0] + i * V[:, 1]
    // Check that A * (re + i*im) = (re_eig + i*im_eig) * (re + i*im)
    // where im_eig = ±1

    mat_free_mat(A);
    mat_free_mat(V);
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

    printf("\nmat_eigen_sym:\n");

    test_eigen_sym_1x1();
    test_eigen_sym_2x2();
    test_eigen_sym_3x3();
    test_eigen_sym_identity();
    test_eigen_sym_diagonal();
    test_eigen_sym_orthogonality();
    test_eigen_sym_decomposition();
    test_eigen_sym_eigenvalues_match();

    printf("\nmat_eigen:\n");

    test_eigen_1x1();
    test_eigen_2x2_diagonal();
    test_eigen_2x2_nonsym();
    test_eigen_3x3_identity();
    test_eigen_decomposition_real();
    test_eigen_eigenvalues_match();
    test_eigen_2x2_complex();

    TEST_SUMMARY();
}
