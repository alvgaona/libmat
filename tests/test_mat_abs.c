#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_abs_negative(void) {
    TEST_BEGIN("mat_abs all negative");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){-1, -2, -3, -4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *out = mat_mat(2, 2);

    mat_abs(out, a);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(out);
    TEST_END();
}

void test_mat_abs_positive(void) {
    TEST_BEGIN("mat_abs all positive");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *out = mat_mat(2, 2);

    mat_abs(out, a);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(out);
    TEST_END();
}

void test_mat_abs_mixed(void) {
    TEST_BEGIN("mat_abs mixed");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){-1, 2, -3, 4, -5, 6});
    Mat *expected = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *out = mat_mat(2, 3);

    mat_abs(out, a);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(out);
    TEST_END();
}

void test_mat_abs_zeros(void) {
    TEST_BEGIN("mat_abs zeros");
    Mat *a = mat_zeros(2, 2);
    Mat *expected = mat_zeros(2, 2);
    Mat *out = mat_mat(2, 2);

    mat_abs(out, a);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(out);
    TEST_END();
}

void test_mat_abs_fractional(void) {
    TEST_BEGIN("mat_abs fractional");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){-1.5, 2.5, -0.5, 0.25});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1.5, 2.5, 0.5, 0.25});
    Mat *out = mat_mat(2, 2);

    mat_abs(out, a);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(out);
    TEST_END();
}

int main(void) {
    printf("mat_abs:\n");

    test_mat_abs_negative();
    test_mat_abs_positive();
    test_mat_abs_mixed();
    test_mat_abs_zeros();
    test_mat_abs_fractional();

    TEST_SUMMARY();
}
