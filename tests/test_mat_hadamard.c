#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_rhadamard_basic(void) {
    TEST_BEGIN("mat_rhadamard basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){5, 12, 21, 32});

    Mat *result = mat_rhadamard(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rhadamard_ones(void) {
    TEST_BEGIN("mat_rhadamard with ones");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *ones = mat_ones(2, 2);

    Mat *result = mat_rhadamard(a, ones);
    CHECK(mat_equals(result, a));

    mat_free_mat(a);
    mat_free_mat(ones);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_rhadamard_zeros(void) {
    TEST_BEGIN("mat_rhadamard with zeros");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *zeros = mat_zeros(2, 2);
    Mat *expected = mat_zeros(2, 2);

    Mat *result = mat_rhadamard(a, zeros);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(zeros);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

void test_mat_hadamard_inplace(void) {
    TEST_BEGIN("mat_hadamard inplace");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *b = mat_from(2, 2, (mat_elem_t[]){2, 3, 4, 5});
    Mat *out = mat_mat(2, 2);
    Mat *expected = mat_from(2, 2, (mat_elem_t[]){2, 6, 12, 20});

    mat_hadamard(out, a, b);
    CHECK(mat_equals(out, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_rhadamard_rectangular(void) {
    TEST_BEGIN("mat_rhadamard rectangular");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});
    Mat *b = mat_from(2, 3, (mat_elem_t[]){2, 2, 2, 2, 2, 2});
    Mat *expected = mat_from(2, 3, (mat_elem_t[]){2, 4, 6, 8, 10, 12});

    Mat *result = mat_rhadamard(a, b);
    CHECK(mat_equals(result, expected));

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(expected);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_hadamard:\n");

    test_mat_rhadamard_basic();
    test_mat_rhadamard_ones();
    test_mat_rhadamard_zeros();
    test_mat_hadamard_inplace();
    test_mat_rhadamard_rectangular();

    TEST_SUMMARY();
}
