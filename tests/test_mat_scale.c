#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_rscale_basic(void) {
    TEST_BEGIN("mat_rscale basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){2, 4, 6, 8});

    Mat *result = mat_rscale(a, 2);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rscale_zero(void) {
    TEST_BEGIN("mat_rscale by zero");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_zeros(2, 2);

    Mat *result = mat_rscale(a, 0);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rscale_negative(void) {
    TEST_BEGIN("mat_rscale negative");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){-1, -2, -3, -4});

    Mat *result = mat_rscale(a, -1);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_scale_inplace(void) {
    TEST_BEGIN("mat_scale inplace");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){3, 6, 9, 12});

    mat_scale(a, 3);
    CHECK(mat_equals(a, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_rscale_fractional(void) {
    TEST_BEGIN("mat_rscale fractional");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){2, 4, 6, 8});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    Mat *result = mat_rscale(a, 0.5f);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_scale:\n");

    test_mat_rscale_basic();
    test_mat_rscale_zero();
    test_mat_rscale_negative();
    test_mat_scale_inplace();
    test_mat_rscale_fractional();

    TEST_SUMMARY();
}
