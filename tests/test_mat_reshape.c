#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_reshape_basic(void) {
    TEST_BEGIN("mat_reshape 2x3 -> 3x2");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    mat_reshape(a, 3, 2);
    CHECK(a->rows == 3 && a->cols == 2);
    // Data should be unchanged (row-major order)
    CHECK(a->data[0] == 1 && a->data[5] == 6);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_reshape_to_vector(void) {
    TEST_BEGIN("mat_reshape 2x3 -> 6x1 (column vector)");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    mat_reshape(a, 6, 1);
    CHECK(a->rows == 6 && a->cols == 1);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_reshape_to_row(void) {
    TEST_BEGIN("mat_reshape 2x3 -> 1x6 (row vector)");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    mat_reshape(a, 1, 6);
    CHECK(a->rows == 1 && a->cols == 6);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_rreshape_basic(void) {
    TEST_BEGIN("mat_rreshape creates new matrix");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    Mat *result = mat_rreshape(a, 3, 2);
    CHECK(result->rows == 3 && result->cols == 2);
    // Original unchanged
    CHECK(a->rows == 2 && a->cols == 3);

    mat_free_mat(a);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_reshape_same_size(void) {
    TEST_BEGIN("mat_reshape same dimensions");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    mat_reshape(a, 2, 2);
    CHECK(mat_equals(a, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    TEST_END();
}

int main(void) {
    printf("mat_reshape:\n");

    test_mat_reshape_basic();
    test_mat_reshape_to_vector();
    test_mat_reshape_to_row();
    test_mat_rreshape_basic();
    test_mat_reshape_same_size();

    TEST_SUMMARY();
}
