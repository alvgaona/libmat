#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_rmul_basic(void) {
    TEST_BEGIN("mat_rmul basic 2x2");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){19, 22, 43, 50});

    Mat *result = mat_rmul(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rmul_identity(void) {
    TEST_BEGIN("mat_rmul with identity");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *eye = mat_eye(2);

    Mat *result = mat_rmul(a, eye);
    CHECK(mat_equals(result, a));

    mat_free_mat(a);
    mat_free_mat(eye);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rmul_rectangular(void) {
    TEST_BEGIN("mat_rmul rectangular 2x3 * 3x2");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *b = mat_from(3, 2, (mat_elem_t[]){7, 8, 9, 10, 11, 12});
    // [1 2 3] * [7  8]  = [1*7+2*9+3*11   1*8+2*10+3*12]  = [58  64]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11   4*8+5*10+6*12]    [139 154]
    //           [11 12]
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){58, 64, 139, 154});

    Mat *result = mat_rmul(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_mul_inplace(void) {
    TEST_BEGIN("mat_mul inplace");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *out = mat_mat(2, 2);
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){19, 22, 43, 50});

    mat_mul(out, a, b);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_rmul_vector(void) {
    TEST_BEGIN("mat_rmul matrix * vector");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Vec *v = mat_vec_from(2, (mat_elem_t[]){5, 6});
    // [1 2] * [5] = [1*5+2*6] = [17]
    // [3 4]   [6]   [3*5+4*6]   [39]
    Vec *expected = mat_vec_from(2, (mat_elem_t[]){17, 39});

    Mat *result = mat_rmul(a, v);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(v);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_mul:\n");

    test_mat_rmul_basic();
    test_mat_rmul_identity();
    test_mat_rmul_rectangular();
    test_mat_mul_inplace();
    test_mat_rmul_vector();

    TEST_SUMMARY();
}
