#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_add_scalar_basic(void) {
    TEST_BEGIN("mat_add_scalar basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){11, 12, 13, 14});

    mat_add_scalar(a, 10);
    CHECK(mat_equals(a, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_add_scalar_negative(void) {
    TEST_BEGIN("mat_add_scalar negative");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){10, 20, 30, 40});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){5, 15, 25, 35});

    mat_add_scalar(a, -5);
    CHECK(mat_equals(a, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_add_scalar_zero(void) {
    TEST_BEGIN("mat_add_scalar zero");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    mat_add_scalar(a, 0);
    CHECK(mat_equals(a, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_radd_scalar_basic(void) {
    TEST_BEGIN("mat_radd_scalar basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){6, 7, 8, 9});

    Mat *result = mat_radd_scalar(a, 5);
    CHECK(mat_equals(result, expected));
    // Original unchanged
    CHECK(a->data[0] == 1);

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_radd_scalar_rectangular(void) {
    TEST_BEGIN("mat_radd_scalar rectangular");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *expected = mat_from(2, 3, (mat_elem_t[]){2, 3, 4, 5, 6, 7});

    Mat *result = mat_radd_scalar(a, 1);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_scalar:\n");

    test_mat_add_scalar_basic();
    test_mat_add_scalar_negative();
    test_mat_add_scalar_zero();
    test_mat_radd_scalar_basic();
    test_mat_radd_scalar_rectangular();

    TEST_SUMMARY();
}
