#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_norm_max_basic(void) {
    TEST_BEGIN("mat_norm_max basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, -3, 2, -1});

    mat_elem_t norm = mat_norm_max(a);

    CHECK_FLOAT_EQ(norm, 3);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_max_all_positive(void) {
    TEST_BEGIN("mat_norm_max all positive");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 5, 2, 4});

    mat_elem_t norm = mat_norm_max(a);

    CHECK_FLOAT_EQ(norm, 5);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_max_all_negative(void) {
    TEST_BEGIN("mat_norm_max all negative");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){-1, -7, -2, -4});

    mat_elem_t norm = mat_norm_max(a);

    CHECK_FLOAT_EQ(norm, 7);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_max_single(void) {
    TEST_BEGIN("mat_norm_max 1x1");
    Mat *a = mat_from(1, 1, (mat_elem_t[]){-42});

    mat_elem_t norm = mat_norm_max(a);

    CHECK_FLOAT_EQ(norm, 42);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_fro_identity(void) {
    TEST_BEGIN("mat_norm_fro identity");
    Mat *a = mat_reye(3);

    mat_elem_t norm = mat_norm_fro(a);

    // Frobenius norm of 3x3 identity = sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
    CHECK_FLOAT_EQ(norm, sqrtf(3));

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_fro_basic(void) {
    TEST_BEGIN("mat_norm_fro basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    mat_elem_t norm = mat_norm_fro(a);

    // sqrt(1 + 4 + 9 + 16) = sqrt(30)
    CHECK_FLOAT_EQ(norm, sqrtf(30));

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_fro_negative(void) {
    TEST_BEGIN("mat_norm_fro with negatives");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){-1, 2, -3, 4});

    mat_elem_t norm = mat_norm_fro(a);

    // sqrt(1 + 4 + 9 + 16) = sqrt(30) (same as positive)
    CHECK_FLOAT_EQ(norm, sqrtf(30));

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_fro_ones(void) {
    TEST_BEGIN("mat_norm_fro ones");
    Mat *a = mat_ones(2, 3);

    mat_elem_t norm = mat_norm_fro(a);

    // sqrt(6 * 1^2) = sqrt(6)
    CHECK_FLOAT_EQ(norm, sqrtf(6));

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_p1(void) {
    TEST_BEGIN("mat_norm p=1");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, -2, 3, -4});

    mat_elem_t norm = mat_norm(a, 1);

    // |1| + |-2| + |3| + |-4| = 10
    CHECK_FLOAT_EQ(norm, 10);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm_p2(void) {
    TEST_BEGIN("mat_norm p=2 equals fro");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    mat_elem_t norm_p2 = mat_norm(a, 2);
    mat_elem_t norm_fro = mat_norm_fro(a);

    // p=2 norm should equal Frobenius norm
    CHECK_FLOAT_EQ(norm_p2, norm_fro);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_norm2_wrapper(void) {
    TEST_BEGIN("mat_norm2 wrapper");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    mat_elem_t norm2 = mat_norm2(a);
    mat_elem_t norm_fro = mat_norm_fro(a);

    // mat_norm2 calls mat_norm(a, 2), should equal Frobenius
    CHECK_FLOAT_EQ(norm2, norm_fro);

    mat_free_mat(a);
    TEST_END();
}

int main(void) {
    printf("mat_norm:\n");

    test_mat_norm_max_basic();
    test_mat_norm_max_all_positive();
    test_mat_norm_max_all_negative();
    test_mat_norm_max_single();
    test_mat_norm_fro_identity();
    test_mat_norm_fro_basic();
    test_mat_norm_fro_negative();
    test_mat_norm_fro_ones();
    test_mat_norm_p1();
    test_mat_norm_p2();
    test_mat_norm2_wrapper();

    TEST_SUMMARY();
}
