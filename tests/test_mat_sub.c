#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_rsub_basic(void) {
    TEST_BEGIN("mat_rsub basic 2x2");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){4, 4, 4, 4});

    Mat *result = mat_rsub(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rsub_zeros(void) {
    TEST_BEGIN("mat_rsub with zeros");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *z = mat_zeros(2, 2);

    Mat *result = mat_rsub(a, z);
    CHECK(mat_equals(result, a));

    mat_free_mat(a);
    mat_free_mat(z);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rsub_self(void) {
    TEST_BEGIN("mat_rsub self equals zero");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_zeros(2, 2);

    Mat *result = mat_rsub(a, a);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_sub_inplace(void) {
    TEST_BEGIN("mat_sub inplace");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *out = mat_mat(2, 2);
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){4, 4, 4, 4});

    mat_sub(out, a, b);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_sub_non_square(void) {
    TEST_BEGIN("mat_sub non-square 2x3");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){6, 5, 4, 3, 2, 1});
    Mat *b = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *expected = mat_from(2, 3, (mat_elem_t[]){5, 3, 1, -1, -3, -5});

    Mat *result = mat_rsub(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_sub:\n");

    test_mat_rsub_basic();
    test_mat_rsub_zeros();
    test_mat_rsub_self();
    test_mat_sub_inplace();
    test_mat_sub_non_square();

    TEST_SUMMARY();
}
