#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_radd_basic(void) {
    TEST_BEGIN("mat_radd basic 2x2");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){6, 8, 10, 12});

    Mat *result = mat_radd(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_radd_zeros(void) {
    TEST_BEGIN("mat_radd with zeros");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *z = mat_zeros(2, 2);

    Mat *result = mat_radd(a, z);
    CHECK(mat_equals(result, a));

    mat_free_mat(a);
    mat_free_mat(z);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_radd_negative(void) {
    TEST_BEGIN("mat_radd with negatives");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){-1, -2, -3, -4});
    Mat *expected = mat_zeros(2, 2);

    Mat *result = mat_radd(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_add_inplace(void) {
    TEST_BEGIN("mat_add inplace");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *out = mat_mat(2, 2);
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){6, 8, 10, 12});

    mat_add(out, a, b);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_add_non_square(void) {
    TEST_BEGIN("mat_add non-square 2x3");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *b = mat_from(2, 3, (mat_elem_t[]){6, 5, 4, 3, 2, 1});
    Mat *expected = mat_from(2, 3, (mat_elem_t[]){7, 7, 7, 7, 7, 7});

    Mat *result = mat_radd(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_add:\n");

    test_mat_radd_basic();
    test_mat_radd_zeros();
    test_mat_radd_negative();
    test_mat_add_inplace();
    test_mat_add_non_square();

    TEST_SUMMARY();
}
