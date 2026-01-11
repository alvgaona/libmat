#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_equals_same(void) {
    TEST_BEGIN("mat_equals same matrix");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    CHECK(mat_equals(a, a));

    mat_free_mat(a);
    TEST_END();
}

void test_mat_equals_identical(void) {
    TEST_BEGIN("mat_equals identical matrices");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    CHECK(mat_equals(a, b));

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_equals_different_values(void) {
    TEST_BEGIN("mat_equals different values");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 5});

    CHECK(!mat_equals(a, b));

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_equals_different_dims(void) {
    TEST_BEGIN("mat_equals different dimensions");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    CHECK(!mat_equals(a, b));

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_equals_tol(void) {
    TEST_BEGIN("mat_equals_tol within tolerance");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1.0f, 2.0f, 3.0f, 4.0f});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){1.0001f, 2.0001f, 3.0001f, 4.0001f});

    CHECK(mat_equals_tol(a, b, 0.001f));
    CHECK(!mat_equals_tol(a, b, 0.00001f));

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_equals_zeros(void) {
    TEST_BEGIN("mat_equals two zero matrices");
    Mat *a = mat_zeros(3, 3);
    Mat *b = mat_zeros(3, 3);

    CHECK(mat_equals(a, b));

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

int main(void) {
    printf("mat_equals:\n");

    test_mat_equals_same();
    test_mat_equals_identical();
    test_mat_equals_different_values();
    test_mat_equals_different_dims();
    test_mat_equals_tol();
    test_mat_equals_zeros();

    TEST_SUMMARY();
}
