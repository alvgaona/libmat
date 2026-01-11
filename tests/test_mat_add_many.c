#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_add_many_two(void) {
    TEST_BEGIN("mat_add_many two matrices");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *out = mat_mat(2, 2);
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){6, 8, 10, 12});

    mat_add_many(out, 2, a, b);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_add_many_three(void) {
    TEST_BEGIN("mat_add_many three matrices");
    Mat *a = mat_ones(2, 2);
    Mat *b = mat_ones(2, 2);
    Mat *c = mat_ones(2, 2);
    Mat *out = mat_mat(2, 2);
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){3, 3, 3, 3});

    mat_add_many(out, 3, a, b, c);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(c);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_radd_many_two(void) {
    TEST_BEGIN("mat_radd_many two matrices");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){6, 8, 10, 12});

    Mat *result = mat_radd_many(2, a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_radd_many_four(void) {
    TEST_BEGIN("mat_radd_many four matrices");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 0, 0, 0});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){0, 2, 0, 0});
    Mat *c = mat_from(2, 2, (mat_elem_t[]){0, 0, 3, 0});
    Mat *d = mat_from(2, 2, (mat_elem_t[]){0, 0, 0, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    Mat *result = mat_radd_many(4, a, b, c, d);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(c);
    mat_free_mat(d);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_add_many_single(void) {
    TEST_BEGIN("mat_add_many single matrix");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *out = mat_mat(2, 2);

    mat_add_many(out, 1, a);
    CHECK(mat_equals(out, a));

    mat_free_mat(a);
    mat_free_mat(out);
    TEST_END();
}

int main(void) {
    printf("mat_add_many:\n");

    test_mat_add_many_two();
    test_mat_add_many_three();
    test_mat_radd_many_two();
    test_mat_radd_many_four();
    test_mat_add_many_single();

    TEST_SUMMARY();
}
