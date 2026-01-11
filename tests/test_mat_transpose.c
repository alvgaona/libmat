#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_rt_square(void) {
    TEST_BEGIN("mat_rt square 2x2");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 3, 2, 4});

    Mat *result = mat_rt(a);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rt_rectangular(void) {
    TEST_BEGIN("mat_rt rectangular 2x3 -> 3x2");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *expected = mat_from(3, 2, (mat_elem_t[]){1, 4, 2, 5, 3, 6});

    Mat *result = mat_rt(a);
    CHECK(mat_equals(result, expected));
    CHECK(result->rows == 3 && result->cols == 2);

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rt_identity(void) {
    TEST_BEGIN("mat_rt identity unchanged");
    Mat *eye = mat_reye(3);

    Mat *result = mat_rt(eye);
    CHECK(mat_equals(result, eye));

    mat_free_mat(eye);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rt_double_transpose(void) {
    TEST_BEGIN("mat_rt double transpose equals original");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    Mat *once = mat_rt(a);
    Mat *twice = mat_rt(once);
    CHECK(mat_equals(twice, a));

    mat_free_mat(a);
    mat_free_mat(once);
    mat_free_mat(twice);
    TEST_END();
}

void test_mat_t_inplace(void) {
    TEST_BEGIN("mat_t inplace");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *out = mat_mat(3, 2);
    Mat *expected = mat_from(3, 2, (mat_elem_t[]){1, 4, 2, 5, 3, 6});

    mat_t(out, a);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_rt_vector(void) {
    TEST_BEGIN("mat_rt column vector -> row vector");
    Vec *v = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});
    Mat *expected = mat_from(1, 3, (mat_elem_t[]){1, 2, 3});

    Mat *result = mat_rt(v);
    CHECK(mat_equals(result, expected));
    CHECK(result->rows == 1 && result->cols == 3);

    mat_free_mat(v);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_transpose:\n");

    test_mat_rt_square();
    test_mat_rt_rectangular();
    test_mat_rt_identity();
    test_mat_rt_double_transpose();
    test_mat_t_inplace();
    test_mat_rt_vector();

    TEST_SUMMARY();
}
